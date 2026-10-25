"""The scheduler against mlx-lm >= 0.32 (token-level stop sequences + a text state machine).

mlx-lm 0.31 matched <think>/<tool_call> as token sequences inside the scheduler; 0.32 strips them
from decoded text in the HTTP handler, and the scheduler only reports stops. These tests pin what
rec's responses look like under that split, with a scripted decoder and no model.
"""

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from rec_mlx.server import make_server
from test_serving import Engine, Tokenizer


class MarkerTokenizer(Tokenizer):
    eos_token_id = 7  # the tokenizer's own EOS is not one of the checkpoint's

    def get_vocab(self):
        return {"<think>": 1000, "</think>": 1001}


EOS = (0, "")  # the checkpoint's end-of-sequence token
CHARS = lambda text: [(ord(c), c) for c in text]  # noqa: E731  one token per character


class ScriptedEngine(Engine):
    """Emits its script like the real decoder: it stops when asked to (should_stop), not by itself."""

    tokenizer = MarkerTokenizer()
    eos = {0}

    def __init__(self, script):
        super().__init__()
        self.script = script
        self.emitted = 0

    def generate_tokens(self, ids, sampling, *, should_stop, return_logprobs=False, grammar=None):
        for event in self.script:
            if should_stop():
                break
            token, text = event if isinstance(event, tuple) else (ord("x"), event)
            self.emitted += 1
            yield {"token": token, "text": text, "finish_reason": None}
        yield {
            "metrics": {
                "prompt_tokens": len(ids),
                "cached_prompt_tokens": 1,
                "generated_tokens": self.emitted,
            }
        }


class HostMlxLmTest(unittest.TestCase):
    def serve(self, script):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.engine = ScriptedEngine(script)
        server = make_server(self.engine, ("127.0.0.1", 0), Path(temp.name) / "metrics")
        thread = threading.Thread(target=server.serve_forever)
        thread.start()

        def stop():
            server.generator.cancel()
            server.shutdown()
            server.generator.join()
            server.server_close()
            thread.join()

        self.addCleanup(stop)
        return server

    def post(self, server, **body):
        connection = http.client.HTTPConnection(*server.server_address, timeout=10)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                json.dumps({"model": "flint", "messages": [{"role": "user", "content": "go"}], **body}),
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            return response.status, response.read().decode()
        finally:
            connection.close()

    def test_checkpoint_eos_ends_the_generation(self):
        server = self.serve(["a", EOS, "junk"])
        status, body = self.post(server)
        self.assertEqual(status, 200, body)
        data = json.loads(body)
        self.assertEqual(data["choices"][0]["message"]["content"], "a")
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")
        # The scheduler asked the decoder to stop at the EOS token: "junk" was never produced.
        self.assertEqual(self.engine.emitted, 2)
        self.assertEqual(data["usage"]["completion_tokens"], 2)

    def test_the_tokenizers_own_eos_id_is_not_a_stop(self):
        # mlx-lm >= 0.32 adds tokenizer.eos_token_id to the stop set; rec stops on the checkpoint's list.
        server = self.serve(["a", (7, "\x07"), "b", EOS])
        status, body = self.post(server)
        self.assertEqual(status, 200, body)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "a\x07b")

    def test_user_stop_word_ends_the_generation_and_is_stripped(self):
        server = self.serve([*CHARS("aEND"), "junk"])
        status, body = self.post(server, grammar="none", stop=["END"])
        self.assertEqual(status, 200, body)
        choice = json.loads(body)["choices"][0]
        self.assertEqual(choice["message"]["content"], "a")
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(self.engine.emitted, 4)

    def test_reasoning_is_split_from_the_answer_and_the_stop_token_is_stripped(self):
        # Markers arrive split across decode steps, as a real tokenizer produces them.
        server = self.serve(["<th", "ink>", "pla", "n</thi", "nk>", "ans", "wer", EOS])
        status, body = self.post(server)
        self.assertEqual(status, 200, body)
        data = json.loads(body)
        choice = data["choices"][0]
        self.assertEqual(choice["message"]["content"], "answer")
        self.assertEqual(choice["message"]["reasoning"], "plan")
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(data["usage"]["completion_tokens"], 8)
        self.assertEqual(data["usage"]["prompt_tokens_details"]["cached_tokens"], 1)

    def test_streamed_deltas_carry_the_same_split(self):
        server = self.serve(["<think>", "plan", "</think>", "ans", "wer", EOS])
        status, body = self.post(server, stream=True, stream_options={"include_usage": True})
        self.assertEqual(status, 200, body)
        events = [
            json.loads(line[6:])
            for line in body.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        deltas = [choice["delta"] for event in events for choice in event["choices"]]
        self.assertEqual("".join(d.get("reasoning") or "" for d in deltas), "plan")
        self.assertEqual("".join(d.get("content") or "" for d in deltas), "answer")
        self.assertEqual(events[-1]["usage"]["prompt_tokens_details"]["cached_tokens"], 1)

    def test_tool_call_is_parsed_from_the_text_between_the_markers(self):
        call = "<tool_call>get<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
        server = self.serve(["<think>", "x", "</think>", call[:9], call[9:], EOS])
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        status, body = self.post(server, tools=tools, grammar="none")
        self.assertEqual(status, 200, body)
        choice = json.loads(body)["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        (made,) = choice["message"]["tool_calls"]
        self.assertEqual(made["function"]["name"], "get")
        self.assertEqual(json.loads(made["function"]["arguments"]), {"city": "Paris"})
        self.assertIsNone(choice["message"]["content"])


if __name__ == "__main__":
    unittest.main()
