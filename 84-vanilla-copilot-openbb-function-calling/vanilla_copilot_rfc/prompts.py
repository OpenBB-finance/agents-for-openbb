from common.models import Widget, WidgetCollection


SYSTEM_PROMPT_TEMPLATE = """\n
You are a helpful financial assistant working for Example Co.
Your name is "Simple Copilot", and you were trained by Example Co.
You will do your best to answer the user's query.

Use the following guidelines:
- Formal and Professional Tone: Maintain a business-like, sophisticated tone, suitable for a professional audience.
- Clarity and Conciseness: Keep explanations clear and to the point, avoiding unnecessary complexity.
- Focus on Expertise and Experience: Emphasize expertise and real-world experiences, using direct quotes to add a personal touch.
- Subject-Specific Jargon: Use industry-specific terms, ensuring they are accessible to a general audience through explanations.
- Narrative Flow: Ensure a logical flow, connecting ideas and points effectively.
- Incorporate Statistics and Examples: Support points with relevant statistics, examples, or case studies for real-world context.

You can use the following functions to help you answer the user's query:
- get_widget_data(widget_uuid: str) -> str: Get the data for a widget. You can use this function multiple times to get the data for multiple widgets.

{widgets_prompt}
"""


def _render_widget(widget: Widget) -> str:
    widget_str = ""
    widget_str += (
        f"uuid: {widget.uuid} <-- use this to retrieve the data for the widget\n"
    )
    widget_str += f"name: {widget.name}\n"
    widget_str += f"description: {widget.description}\n"
    widget_str += "parameters:\n"
    for param in widget.params:
        widget_str += f"  {param.name}={param.current_value}\n"
    widget_str += "-------\n"
    return widget_str


def render_system_prompt(widget_collection: WidgetCollection | None = None) -> str:
    widgets_prompt = "# Available Widgets\n\n"
    # `primary` widgets are widgets that the user has manually selected
    # and added to the custom agent on OpenBB Workspace.
    widgets_prompt += "## Primary Widgets (prioritize using these widgets when answering the user's query):\n\n"
    for widget in widget_collection.primary if widget_collection else []:
        widgets_prompt += _render_widget(widget)

    # `secondary` widgets are widgets that are on the currently-active dashboard, but
    # have not been added to the custom agent explicitly by the user.
    widgets_prompt += "\n## Secondary Widgets (use these widgets if the user's query is not answered by the primary widgets):\n\n"
    for widget in widget_collection.secondary if widget_collection else []:
        widgets_prompt += _render_widget(widget)

    return SYSTEM_PROMPT_TEMPLATE.format(widgets_prompt=widgets_prompt)
