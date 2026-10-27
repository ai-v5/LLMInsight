"""HTTP regressions using synthetic data and an isolated loopback server."""
from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from llminsight.server import app


class ServerSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="_security_", dir=Path(__file__).parents[1] / "scripts")
        base = Path(cls.tmp.name)
        cls.web = base / "web"
        cls.web.mkdir()
        (cls.web / "index.html").write_text("synthetic-index", encoding="utf-8")
        (cls.web / "app.js").write_text("// synthetic asset", encoding="utf-8")
        sibling = base / "web_backup"
        sibling.mkdir()
        (sibling / "private.txt").write_text("synthetic-private-marker", encoding="utf-8")
        cls.web_patch = patch.object(app, "WEB_DIR", str(cls.web))
        cls.web_patch.start()
        cls.server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        cls.authority = f"127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.web_patch.stop()
        cls.tmp.cleanup()

    def request(self, path="/", method="GET", headers=None, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def post_headers(self, **extra):
        return {"Content-Type": "application/json", "X-LLMInsight-Request": "1", **extra}

    def test_direct_loopback_get_and_localhost_alias_work(self):
        self.assertEqual(self.request()[0], 200)
        self.assertEqual(self.request(headers={"Host": f"localhost:{self.server.server_port}"})[0], 200)
        self.assertEqual(self.request("/app.js")[1], b"// synthetic asset")

    def test_foreign_host_and_port_are_rejected(self):
        for host in (f"audit.invalid:{self.server.server_port}", "127.0.0.1:1", self.authority + ".invalid"):
            with self.subTest(host=host):
                self.assertEqual(self.request(headers={"Host": host})[0], 403)

    def test_foreign_null_and_different_port_origins_are_rejected(self):
        for origin in ("https://audit.invalid", "null", "http://127.0.0.1:1"):
            with self.subTest(origin=origin):
                self.assertEqual(self.request(headers={"Origin": origin})[0], 403)
        self.assertEqual(self.request(headers={"Origin": f"http://{self.authority}"})[0], 200)

    def test_cross_site_and_cross_origin_same_site_requests_are_rejected(self):
        for site in ("cross-site", "same-site"):
            self.assertEqual(self.request(headers={"Sec-Fetch-Site": site})[0], 403)

    def test_get_and_head_never_trigger_llm(self):
        with patch.object(app.STATE, "ready", True), patch.dict(app.ROUTES, {"/api/llm": lambda: {"called": True}}):
            self.assertEqual(self.request("/api/llm")[0], 405)
            self.assertEqual(self.request("/api/llm", method="HEAD")[0], 405)
            status, data = self.request("/api/llm", "POST", self.post_headers(), b"{}")
            self.assertEqual(status, 200)
            self.assertTrue(json.loads(data)["called"])

    def test_post_requires_non_simple_json_request(self):
        with patch.object(app.STATE, "reset", return_value={"ok": True}) as reset:
            self.assertEqual(self.request("/api/reset", "POST", {"Content-Type": "text/plain"}, b"{}")[0], 403)
            self.assertEqual(self.request("/api/reset", "POST", self.post_headers(**{"Content-Type": "text/plain"}), b"{}")[0], 415)
            self.assertEqual(self.request("/api/reset", "POST", self.post_headers(Origin="https://audit.invalid"), b"{}")[0], 403)
            reset.assert_not_called()
            self.assertEqual(self.request("/api/reset", "POST", self.post_headers(), b"{}")[0], 200)
            reset.assert_called_once()

    def test_valid_json_load_reaches_handler(self):
        with patch.object(app.STATE, "start_load", return_value={"ok": True}) as load:
            status, _ = self.request("/api/load", "POST", self.post_headers(Origin=f"http://{self.authority}"), b'{"dir":"synthetic-profile"}')
            self.assertEqual(status, 200)
            load.assert_called_once_with("synthetic-profile")

    def test_post_body_is_consumed_before_reusing_connection(self):
        with patch.object(app.STATE, "ready", True), patch.object(app.STATE, "reset", return_value={"ok": True}), patch.dict(app.ROUTES, {"/api/llm": lambda: {"synthetic": True}}):
            for endpoint in ("/api/llm", "/api/reset"):
                with self.subTest(endpoint=endpoint):
                    connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
                    try:
                        connection.request("POST", endpoint, body=b"{}", headers=self.post_headers())
                        response = connection.getresponse()
                        self.assertEqual(response.status, 200)
                        response.read()
                        connection.request("GET", "/app.js")
                        response = connection.getresponse()
                        self.assertEqual(response.status, 200)
                        self.assertEqual(response.read(), b"// synthetic asset")
                    finally:
                        connection.close()

    def test_invalid_json_objects_do_not_reach_mutation(self):
        with patch.object(app.STATE, "reset", return_value={"ok": True}) as reset:
            for body in (b"not-json", b"[]", b"null"):
                self.assertEqual(self.request("/api/reset", "POST", self.post_headers(), body)[0], 400)
            reset.assert_not_called()

    def test_static_files_cannot_escape_to_sibling_with_same_prefix(self):
        for path in ("/../web_backup/private.txt", "/..%2fweb_backup/private.txt", "/..%5cweb_backup/private.txt"):
            with self.subTest(path=path):
                status, data = self.request(path)
                # Backslashes are path separators on Windows, the supported host.
                if "5c" in path and app.os.name != "nt":
                    continue
                self.assertEqual(status, 403)
                self.assertNotIn(b"synthetic-private-marker", data)

    def test_static_symlinks_cannot_escape(self):
        link = self.web / "outside.txt"
        try:
            link.symlink_to(self.web.parent / "web_backup" / "private.txt")
        except OSError:
            self.skipTest("symlink creation is unavailable on this host")
        try:
            self.assertEqual(self.request("/outside.txt")[0], 403)
        finally:
            link.unlink()


if __name__ == "__main__":
    unittest.main()
