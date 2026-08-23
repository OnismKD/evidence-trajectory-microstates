"""Small in-memory example; equivalent to ``evidence-microstates demo``."""

from evidence_microstates.cli import demo_command

if __name__ == "__main__":
    demo_command("outputs/demo", random_state=42)
