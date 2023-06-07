import numpy as np
import dataclasses as dc


@dc.dataclass
class CtrlArguments:
    train_data: str = dc.field(
        default="data/training_cunique_with_distractors.json",
        metadata={"help": "A CSV list of training data files"}
    )

    formulation: str = dc.field(
        default="areg_ltr",
        metadata={"help": "Type of problem definition: autoregressive (areg) or u-PMLM (upmlm) or mixed (if predict_questions is set)"}
    )

    context_strategy: str = dc.field(
        default="take_first",
        metadata={"help": "How to deal with contexts greater than a specified length"}
    )

    tokenizer_file: str = dc.field(
        default="tokenizer.json",
        metadata={"help": "A JSON file (in the format provided by HuggingFace's tokenizers library) with a trained tokenizer"}
    )

    sequence_length: int = dc.field(
        default=256,
        metadata={"help": "The max sequence length"}
    )

    force_prepend_control: bool = dc.field(
        default=False,
        metadata={"help": "If the control code should be prepended for all sliding windows. Otherwise, it is only prepended at the start of the sequence"}
    )

