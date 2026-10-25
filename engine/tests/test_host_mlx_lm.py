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


class ScriptedEngine(Engine):
    tokenizer = MarkerTokenizer()
    eos = {0}

    def __init__(self, pieces):
        super().__init__()
        self.pieces = pieces

    def generate_tokens(self, ids, sampling, *, should_stop, return_logprobs=False, grammar=None):
        count = 0
        for piece in self.pieces:
            count += 1
            token = 0 if piece is None else ord("x")
            yield {
                "token": token,
                "text": "" if piece is None else piece,
                "finish_reason": None,
            }
            if piece is None:
                break
        yield {
            "metrics": {
                "prompt_tokens": len(ids),
                "cached_prompt_tokens": 1,
                "generated_tokens": count,
            }
        }


class HostMlxLmTest(unittest.TestCase):
    def serve(self, pieces):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.engine = ScriptedEngine(pieces)
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

    def test_tokenizer_stops_on_the_checkpoint_eos_list_only(self):
        server = self.serve([None])
        self.assertEqual(server.generator.tokenizer.eos_token_ids, {0})

    def test_reasoning_is_split_from_the_answer_and_the_stop_token_is_stripped(self):
        # Markers arrive split across decode steps, as a real tokenizer produces them.
        server = self.serve(["<th", "ink>", "pla", "n</thi", "nk>", "ans", "wer", None])
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
        server = self.serve(["<think>", "plan", "</think>", "ans", "wer", None])
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
        server = self.serve(["<think>", "x", "</think>", call[:9], call[9:], None])
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
