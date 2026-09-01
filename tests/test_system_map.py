import importlib.util
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).parents[1]
MAP_PATH = ROOT / "data" / "system-map.v1.json"
VALIDATOR = ROOT / "scripts" / "validate-system-map.py"

spec = importlib.util.spec_from_file_location("system_map_validator", VALIDATOR)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)


class SystemMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system_map = json.loads(MAP_PATH.read_text())

    def test_live_system_map_is_valid(self):
        self.assertEqual(validator.validate_system_map(self.system_map), [])

    def test_missing_authority_fails_closed(self):
        broken = json.loads(json.dumps(self.system_map))
        broken["authorities"]["missing"] = "docs/DOES-NOT-EXIST.md"
        errors = validator.validate_system_map(broken)
        self.assertIn("missing referenced path: docs/DOES-NOT-EXIST.md", errors)

    def test_control_graph_has_all_intents_and_layers(self):
        self.assertEqual(set(self.system_map["intents"]), {"read", "reproduce", "publish", "triage"})
        self.assertEqual([layer["id"] for layer in self.system_map["layers"]], ["L0", "L1", "L2", "L3", "L4"])


if __name__ == "__main__":
    unittest.main()
