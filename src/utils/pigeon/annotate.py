import random
import functools
from IPython.display import display, clear_output
from ipywidgets import (
    Button,
    Dropdown,
    HTML,
    Box,
    IntSlider,
    FloatSlider,
    Textarea,
    Output,
    Layout,
    ToggleButtons,
)

# The whole script is from the pigeon package: https://github.com/agermanidis/pigeon
# Slightly modified for our needs
def annotate(
    examples, options=None, shuffle=False, include_skip=True, display_fn=display
):
    """
    Build an interactive widget for annotating a list of input examples.

    Parameters
    ----------
    examples: list(any), list of items to annotate
    options: list(any) or tuple(start, end, [step]) or None
             if list: list of labels for binary classification task (Dropdown or Buttons)
             if tuple: range for regression task (IntSlider or FloatSlider)
             if None: arbitrary text input (TextArea)
    shuffle: bool, shuffle the examples before annotating
    include_skip: bool, include option to skip example while annotating
    display_fn: func, function for displaying an example to the user

    Returns
    -------
    annotations : list of tuples, list of annotated examples (example, label)
    """
    examples = list(examples)
    if shuffle:
        random.shuffle(examples)

    annotations = []
    certainties = []
    current_index = -1

    def set_label_text():
        nonlocal count_label
        count_label.value = "{} examples annotated, {} examples left".format(
            len(annotations), len(examples) - current_index
        )

    def show_next():
        nonlocal current_index
        current_index += 1
        set_label_text()
        certainty.value = ""
        if current_index >= len(examples):
            for btn in buttons:
                btn.disabled = True
            print("Annotation done.")
            return
        with out:
            clear_output(wait=True)
            display_fn(examples[current_index])

    def add_annotation(annotation):
        annotations.append((examples[current_index], annotation))
        certainties.append((examples[current_index], certainty.value))
        show_next()

    def skip(btn):
        show_next()

    count_label = HTML()
    set_label_text()
    display(count_label)

    if type(options) == list:
        task_type = "classification"
    elif type(options) == tuple and len(options) in [2, 3]:
        task_type = "regression"
    elif options is None:
        task_type = "captioning"
    else:
        raise Exception("Invalid options")

    buttons = []

    if task_type == "classification":
        use_dropdown = len(options) > 15

        if use_dropdown:
            dd = Dropdown(options=options)
            display(dd)
            btn = Button(description="submit")

            def on_click(btn):
                add_annotation(dd.value)

            btn.on_click(on_click)
            buttons.append(btn)

        else:
            for label in options:
                btn = Button(
                    description=label, layout=Layout(width="300px", height="60px")
                )

                def on_click(label, btn):
                    if certainty.value == "":
                        return
                    add_annotation(label)

                btn.on_click(functools.partial(on_click, label))
                buttons.append(btn)

    elif task_type == "regression":
        target_type = type(options[0])
        if target_type == int:
            cls = IntSlider
        else:
            cls = FloatSlider
        if len(options) == 2:
            min_val, max_val = options
            slider = cls(min=min_val, max=max_val)
        else:
            min_val, max_val, step_val = options
            slider = cls(min=min_val, max=max_val, step=step_val)
        display(slider)
        btn = Button(description="submit")

        def on_click(btn):
            add_annotation(slider.value)

        btn.on_click(on_click)
        buttons.append(btn)

    else:
        ta = Textarea()
        display(ta)
        btn = Button(description="submit")

        def on_click(btn):
            add_annotation(ta.value)

        btn.on_click(on_click)
        buttons.append(btn)

    def on_click(label, btn):
        if certainty.value == "":
            return
        add_annotation(label)

    # DUPLICATE BUTTON
    btn = Button(
        description="Duplicate",
        layout=Layout(width="300px", height="60px"),
        button_style="info",
    )

    label = "Duplicate"
    btn.on_click(functools.partial(on_click, label))
    buttons.append(btn)

    # SKIP BUTTON
    btn = Button(
        description="Skip",
        layout=Layout(width="300px", height="60px"),
        button_style="danger",
    )

    def on_click(label, btn):
        certainty.value = ""
        add_annotation(label)

    label = "Skipped"
    btn.on_click(functools.partial(on_click, label))
    buttons.append(btn)

    # Let's show it all
    box_layout = Layout(display="flex", flex_flow="wrap")

    box = Box(buttons, layout=box_layout)

    certainty = ToggleButtons(
        options=["", "Very Uncertain", "Uncertain", "Certain", "Very Certain"],
        description="Certainty:",
        disabled=False,
        button_style="info",  # 'success', 'info', 'warning', 'danger' or ''
    )
    out = Output()
    display(out)

    display(certainty)

    display(box)

    show_next()

    return annotations, certainties
