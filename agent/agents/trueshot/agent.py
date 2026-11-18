"""ADK app assembly: workflow + plugins + resumability. Loaded by the ADK server as app `trueshot`."""

from google.adk.apps import App, ResumabilityConfig

from .graph import build_workflow
from .plugins import all_plugins

root_agent = build_workflow()

app = App(
    name="trueshot",
    root_agent=root_agent,
    plugins=all_plugins(),
    # Lets a run continue after a disconnect or restart (and after the approval pause).
    resumability_config=ResumabilityConfig(is_resumable=True),
)
