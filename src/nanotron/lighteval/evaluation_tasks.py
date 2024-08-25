# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval

This file generally create just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
Edit this file to add your own task if needed
"""
import re
from dataclasses import asdict
from typing import Dict, List, Tuple

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.default_prompts import LETTER_INDICES

_TASKS_STRINGS: List[Tuple[LightevalTaskConfig, str]] = []
_TASKS: List[LightevalTaskConfig] = []

## COMMON_SENSE_REASONING_TASKS ##
COMMON_SENSE_REASONING_TASKS = [
    LightevalTaskConfig(
        name="hellaswag",
        prompt_function="hellaswag_prompt",
        hf_repo="hellaswag",
        hf_subset="default",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="winogrande",
        prompt_function="winogrande",
        hf_repo="winogrande",
        hf_subset="winogrande_xl",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="piqa",
        prompt_function="piqa_harness",
        hf_repo="piqa",
        hf_subset="plain_text",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="siqa",
        prompt_function="siqa_prompt",
        hf_repo="lighteval/siqa",
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="openbookqa",
        prompt_function="openbookqa",
        hf_repo="openbookqa",
        hf_subset="main",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="arc:easy",
        prompt_function="arc",
        hf_repo="ai2_arc",
        hf_subset="ARC-Easy",
        evaluation_splits=["test"],
        generation_size=1,
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="arc:challenge",
        prompt_function="arc",
        hf_repo="ai2_arc",
        hf_subset="ARC-Challenge",
        evaluation_splits=["test"],
        generation_size=1,
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="commonsense_qa",
        prompt_function="commonsense_qa_prompt",
        hf_repo="commonsense_qa",
        hf_subset="default",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
]


def commonsense_qa_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=LETTER_INDICES.index(line["answerKey"].strip()),
        instruction="",
    )


def siqa_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["context"] + " " + line["question"],
        choices=[f" {c}" for c in [line["answerA"], line["answerB"], line["answerC"]]],
        gold_index=int(line["label"]) - 1,
        instruction="",
    )


def hellaswag_prompt(line, task_name: str = None):
    def preprocess(text):
        """Comes from AiHarness"""
        # text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    ctx = f"{line['ctx_a']} {line['ctx_b'].capitalize()} "
    return Doc(
        task_name=task_name,
        query=preprocess(line["activity_label"] + ": " + ctx),
        choices=[" " + preprocess(ending) for ending in line["endings"]],
        gold_index=int(line["label"]) if line["label"] != "" else -1,  # -1 for test
        # "metric": "choices_loglikelihood",
    )


# 0 short for common sense
COMMON_SENSE_REASONING_STRING = [(t, f"custom|{t.name}|0|1") for t in COMMON_SENSE_REASONING_TASKS]
_TASKS_STRINGS.extend(COMMON_SENSE_REASONING_STRING)
_TASKS += COMMON_SENSE_REASONING_TASKS

## WORLD_KNOWLEDGE_TASKS ##

WORLD_KNOWLEDGE_TASKS = [
    LightevalTaskConfig(
        name="trivia_qa",
        prompt_function="triviaqa",
        hf_repo="trivia_qa",
        hf_subset="rc.nocontext",
        metric=[Metrics.quasi_exact_match],
        generation_size=20,
        stop_sequence=["\n", ".", ","],
    ),
    LightevalTaskConfig(
        name="natural_questions",
        prompt_function="natural_questions_prompt",
        hf_repo="lighteval/natural_questions_clean",
        hf_subset="default",
        metric=[Metrics.quasi_exact_match],
        generation_size=20,
        stop_sequence=["\n", ".", ","],
    ),
]


def natural_questions_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["question"] + "?\nAnswer: ",
        choices=[line["short_answers"]],
        gold_index=0,
        instruction="",
    )


WORLD_KNOWLEDGE_STRING = [(t, f"custom|{t.name}|5|1") for t in WORLD_KNOWLEDGE_TASKS]
# WORLD_KNOWLEDGE_STRING = {t: f'custom|{t.name}|0|1' for t in WORLD_KNOWLEDGE_TASKS}
_TASKS_STRINGS.extend(WORLD_KNOWLEDGE_STRING)
_TASKS += WORLD_KNOWLEDGE_TASKS

## Reading comprehension ##

READING_COMP_TASKS = [
    LightevalTaskConfig(
        name="super_glue:boolq",
        prompt_function="boolq_prompt",
        hf_repo="super_glue",
        hf_subset="boolq",
        metric=["target_perplexity"],
    ),
    LightevalTaskConfig(
        name="quac",
        prompt_function="quac",
        hf_repo="lighteval/quac_helm",
        hf_subset="deault",
        metric=[Metrics.quasi_exact_match],
        generation_size=20,
        stop_sequence=["\n", ".", ","],
    ),
]


def boolq_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['passage']}\nQuestion: {line['question'].capitalize()}?\nAnswer:",
        choices=[" No", " Yes"],  # Only gold
        gold_index=int(line["label"]),
    )


READING_COMP_STRING = [(t, f"custom|{t.name}|0|1") for t in READING_COMP_TASKS]
_TASKS_STRINGS.extend(READING_COMP_STRING)
_TASKS += READING_COMP_TASKS


## MATH ##
class CustomMathEvaluationTask(LightevalTaskConfig):
    """Custom class for math tasks with all the defaults set"""

    def __init__(
        self,
        name,
        prompt_function="math",
        hf_repo="lighteval/MATH",
        hf_subset=None,
        metric=[Metrics.quasi_exact_match_math],
        hf_avail_splits=None,
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        suite=["custom"],
        generation_size=40,
        stop_sequence=None,
        output_regex=None,
        frozen=False,
    ):
        super().__init__(
            name=name,
            prompt_function=prompt_function,
            hf_repo=hf_repo,
            hf_subset=hf_subset,
            metric=metric,
            hf_avail_splits=hf_avail_splits,
            evaluation_splits=evaluation_splits,
            few_shots_split=few_shots_split,
            few_shots_select=few_shots_select,
            suite=suite,
            generation_size=generation_size,
            stop_sequence=stop_sequence,
            output_regex=output_regex,
            frozen=frozen,
        )


MATH_TASKS = [
    CustomMathEvaluationTask(name="math:algebra", hf_subset="algebra"),
    CustomMathEvaluationTask(name="math:counting_and_probability", hf_subset="counting_and_probability"),
    CustomMathEvaluationTask(name="math:geometry", hf_subset="geometry"),
    CustomMathEvaluationTask(name="math:intermediate_algebra", hf_subset="intermediate_algebra"),
    CustomMathEvaluationTask(name="math:number_theory", hf_subset="number_theory"),
    CustomMathEvaluationTask(name="math:prealgebra", hf_subset="prealgebra"),
    CustomMathEvaluationTask(name="math:precalculus", hf_subset="precalculus"),
]
GSM8K = LightevalTaskConfig(
    name="gsm8k",
    prompt_function="gsm8k",
    hf_repo="gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    metric=[Metrics.perfect_exact_match],
    generation_size=10,
    stop_sequence=["\n"],
)


MATH_STRING = [(t, f"custom|{t.name}|4|1") for t in MATH_TASKS]
GSM8K_STRING = [(GSM8K, f"custom|{GSM8K.name}|8|1")]
_TASKS_STRINGS.extend(MATH_STRING)
_TASKS_STRINGS.extend(GSM8K_STRING)
_TASKS += MATH_TASKS + [GSM8K]


## MMLU ##
class CustomMMLUEvaluationTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        prompt_function="mmlu_prompt",
        hf_repo="lighteval/mmlu",
        hf_subset=None,
        #  metric=[Metrics.loglikelihood_acc_single_token],
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        hf_avail_splits=None,
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select=None,
        suite=None,
        generation_size=-1,
        stop_sequence=None,
        output_regex=None,
        frozen=False,
    ):
        super().__init__(
            name=name,
            prompt_function=prompt_function,
            hf_repo=hf_repo,
            hf_subset=hf_subset,
            metric=metric,
            hf_avail_splits=hf_avail_splits,
            evaluation_splits=evaluation_splits,
            few_shots_split=few_shots_split,
            few_shots_select=few_shots_select,
            suite=suite,
            generation_size=generation_size,
            stop_sequence=stop_sequence,
            output_regex=output_regex,
            frozen=frozen,
        )


MMLU_TASKS = [
    CustomMMLUEvaluationTask(name="mmlu:abstract_algebra", hf_subset="abstract_algebra"),
    CustomMMLUEvaluationTask(name="mmlu:anatomy", hf_subset="anatomy"),
    CustomMMLUEvaluationTask(name="mmlu:astronomy", hf_subset="astronomy"),
    CustomMMLUEvaluationTask(name="mmlu:business_ethics", hf_subset="business_ethics"),
    CustomMMLUEvaluationTask(name="mmlu:clinical_knowledge", hf_subset="clinical_knowledge"),
    CustomMMLUEvaluationTask(name="mmlu:college_biology", hf_subset="college_biology"),
    CustomMMLUEvaluationTask(name="mmlu:college_chemistry", hf_subset="college_chemistry"),
    CustomMMLUEvaluationTask(name="mmlu:college_computer_science", hf_subset="college_computer_science"),
    CustomMMLUEvaluationTask(name="mmlu:college_mathematics", hf_subset="college_mathematics"),
    CustomMMLUEvaluationTask(name="mmlu:college_medicine", hf_subset="college_medicine"),
    CustomMMLUEvaluationTask(name="mmlu:college_physics", hf_subset="college_physics"),
    CustomMMLUEvaluationTask(name="mmlu:computer_security", hf_subset="computer_security"),
    CustomMMLUEvaluationTask(name="mmlu:conceptual_physics", hf_subset="conceptual_physics"),
    CustomMMLUEvaluationTask(name="mmlu:econometrics", hf_subset="econometrics"),
    CustomMMLUEvaluationTask(name="mmlu:electrical_engineering", hf_subset="electrical_engineering"),
    CustomMMLUEvaluationTask(name="mmlu:elementary_mathematics", hf_subset="elementary_mathematics"),
    CustomMMLUEvaluationTask(name="mmlu:formal_logic", hf_subset="formal_logic"),
    CustomMMLUEvaluationTask(name="mmlu:global_facts", hf_subset="global_facts"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_biology", hf_subset="high_school_biology"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_chemistry", hf_subset="high_school_chemistry"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_computer_science", hf_subset="high_school_computer_science"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_european_history", hf_subset="high_school_european_history"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_geography", hf_subset="high_school_geography"),
    CustomMMLUEvaluationTask(
        name="mmlu:high_school_government_and_politics", hf_subset="high_school_government_and_politics"
    ),
    CustomMMLUEvaluationTask(name="mmlu:high_school_macroeconomics", hf_subset="high_school_macroeconomics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_mathematics", hf_subset="high_school_mathematics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_microeconomics", hf_subset="high_school_microeconomics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_physics", hf_subset="high_school_physics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_psychology", hf_subset="high_school_psychology"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_statistics", hf_subset="high_school_statistics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_us_history", hf_subset="high_school_us_history"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_world_history", hf_subset="high_school_world_history"),
    CustomMMLUEvaluationTask(name="mmlu:human_aging", hf_subset="human_aging"),
    CustomMMLUEvaluationTask(name="mmlu:human_sexuality", hf_subset="human_sexuality"),
    CustomMMLUEvaluationTask(name="mmlu:international_law", hf_subset="international_law"),
    CustomMMLUEvaluationTask(name="mmlu:jurisprudence", hf_subset="jurisprudence"),
    CustomMMLUEvaluationTask(name="mmlu:logical_fallacies", hf_subset="logical_fallacies"),
    CustomMMLUEvaluationTask(name="mmlu:machine_learning", hf_subset="machine_learning"),
    CustomMMLUEvaluationTask(name="mmlu:management", hf_subset="management"),
    CustomMMLUEvaluationTask(name="mmlu:marketing", hf_subset="marketing"),
    CustomMMLUEvaluationTask(name="mmlu:medical_genetics", hf_subset="medical_genetics"),
    CustomMMLUEvaluationTask(name="mmlu:miscellaneous", hf_subset="miscellaneous"),
    CustomMMLUEvaluationTask(name="mmlu:moral_disputes", hf_subset="moral_disputes"),
    CustomMMLUEvaluationTask(name="mmlu:moral_scenarios", hf_subset="moral_scenarios"),
    CustomMMLUEvaluationTask(name="mmlu:nutrition", hf_subset="nutrition"),
    CustomMMLUEvaluationTask(name="mmlu:philosophy", hf_subset="philosophy"),
    CustomMMLUEvaluationTask(name="mmlu:prehistory", hf_subset="prehistory"),
    CustomMMLUEvaluationTask(name="mmlu:professional_accounting", hf_subset="professional_accounting"),
    CustomMMLUEvaluationTask(name="mmlu:professional_law", hf_subset="professional_law"),
    CustomMMLUEvaluationTask(name="mmlu:professional_medicine", hf_subset="professional_medicine"),
    CustomMMLUEvaluationTask(name="mmlu:professional_psychology", hf_subset="professional_psychology"),
    CustomMMLUEvaluationTask(name="mmlu:public_relations", hf_subset="public_relations"),
    CustomMMLUEvaluationTask(name="mmlu:security_studies", hf_subset="security_studies"),
    CustomMMLUEvaluationTask(name="mmlu:sociology", hf_subset="sociology"),
    CustomMMLUEvaluationTask(name="mmlu:us_foreign_policy", hf_subset="us_foreign_policy"),
    CustomMMLUEvaluationTask(name="mmlu:virology", hf_subset="virology"),
    CustomMMLUEvaluationTask(name="mmlu:world_religions", hf_subset="world_religions"),
]


def mmlu_harness(line, task_name: str = None):
    topic = line["subject"]
    prompt = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"
    prompt += line["question"] + "\n"
    prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["choices"])])
    prompt += "Answer:"

    gold_ix = LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]
    "__few_shots" in line and line["__few_shots"] is True  # We are adding few shots

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[" A", " B", " C", " D"],
        target_for_fewshot_sorting=[" A", " B", " C", " D"][gold_ix],
        gold_index=gold_ix,
        instruction=f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n",
    )


def mmlu_prompt(line, task_name: str = None):
    """MMLU prompt without letters"""
    topic = line["subject"]
    prompt = f"The following are questions about {topic.replace('_', ' ')}.\nQuestion: "
    prompt += line["question"] + "\nAnswer:"

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {c}" for c in line["choices"]],
        gold_index=line["answer"],
        instruction=f"The following are questions about {topic.replace('_', ' ')}.\n",
    )


# MMLU_STRING = {t: f'custom|{t.name}|5|1' for t in MMLU_TASKS}
MMLU_STRING = [(t, f"custom|{t.name}|0|1") for t in MMLU_TASKS]
_TASKS_STRINGS.extend(MMLU_STRING)
_TASKS += MMLU_TASKS

## BBH ##


class CustomBBHEvaluationTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        prompt_function="bbh_prompt",
        hf_repo="lighteval/big_bench_hard",
        hf_subset=None,
        metric=[Metrics.exact_match],
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_select=None,
        suite=None,
        generation_size=4,
        stop_sequence=None,
        output_regex=None,
        frozen=False,
    ):
        super().__init__(
            name=name,
            prompt_function=prompt_function,
            hf_repo=hf_repo,
            hf_subset=hf_subset,
            metric=metric,
            hf_avail_splits=hf_avail_splits,
            evaluation_splits=evaluation_splits,
            few_shots_split=few_shots_split,
            few_shots_select=few_shots_select,
            suite=suite,
            generation_size=generation_size,
            stop_sequence=stop_sequence,
            output_regex=output_regex,
            frozen=frozen,
        )


BBH_TASKS = [
    CustomBBHEvaluationTask(name="bbh:boolean_expressions", hf_subset="boolean_expressions"),
    CustomBBHEvaluationTask(name="bbh:causal_judgement", hf_subset="causal_judgement"),
    CustomBBHEvaluationTask(name="bbh:date_understanding", hf_subset="date_understanding"),
    CustomBBHEvaluationTask(name="bbh:disambiguation_qa", hf_subset="disambiguation_qa"),
    CustomBBHEvaluationTask(name="bbh:dyck_languages", hf_subset="dyck_languages"),
    CustomBBHEvaluationTask(name="bbh:formal_fallacies", hf_subset="formal_fallacies"),
    CustomBBHEvaluationTask(name="bbh:geometric_shapes", hf_subset="geometric_shapes"),
    CustomBBHEvaluationTask(name="bbh:hyperbaton", hf_subset="hyperbaton"),
    CustomBBHEvaluationTask(name="bbh:logical_deduction_five_objects", hf_subset="logical_deduction_five_objects"),
    CustomBBHEvaluationTask(name="bbh:logical_deduction_seven_objects", hf_subset="logical_deduction_seven_objects"),
    CustomBBHEvaluationTask(name="bbh:logical_deduction_three_objects", hf_subset="logical_deduction_three_objects"),
    CustomBBHEvaluationTask(name="bbh:movie_recommendation", hf_subset="movie_recommendation"),
    CustomBBHEvaluationTask(name="bbh:multistep_arithmetic_two", hf_subset="multistep_arithmetic_two"),
    CustomBBHEvaluationTask(name="bbh:navigate", hf_subset="navigate"),
    CustomBBHEvaluationTask(name="bbh:object_counting", hf_subset="object_counting"),
    CustomBBHEvaluationTask(name="bbh:penguins_in_a_table", hf_subset="penguins_in_a_table"),
    CustomBBHEvaluationTask(name="bbh:reasoning_about_colored_objects", hf_subset="reasoning_about_colored_objects"),
    CustomBBHEvaluationTask(name="bbh:ruin_names", hf_subset="ruin_names"),
    CustomBBHEvaluationTask(
        name="bbh:salient_translation_error_detection", hf_subset="salient_translation_error_detection"
    ),
    CustomBBHEvaluationTask(name="bbh:snarks", hf_subset="snarks"),
    CustomBBHEvaluationTask(name="bbh:sports_understanding", hf_subset="sports_understanding"),
    CustomBBHEvaluationTask(name="bbh:temporal_sequences", hf_subset="temporal_sequences"),
    CustomBBHEvaluationTask(
        name="bbh:tracking_shuffled_objects_five_objects", hf_subset="tracking_shuffled_objects_five_objects"
    ),
    CustomBBHEvaluationTask(
        name="bbh:tracking_shuffled_objects_seven_objects", hf_subset="tracking_shuffled_objects_seven_objects"
    ),
    CustomBBHEvaluationTask(
        name="bbh:tracking_shuffled_objects_three_objects", hf_subset="tracking_shuffled_objects_three_objects"
    ),
    CustomBBHEvaluationTask(name="bbh:web_of_lies", hf_subset="web_of_lies"),
    CustomBBHEvaluationTask(name="bbh:word_sorting", hf_subset="word_sorting"),
]


def bbh_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["input"] + "\nAnswer: ",
        choices=[line["target"]],
        gold_index=0,
    )


# BBH_STRING = {t: f'custom|{t.name}|3|1' for t in BBH_TASKS}
BBH_STRING = [(t, f"custom|{t.name}|0|1") for t in BBH_TASKS]
_TASKS_STRINGS.extend(BBH_STRING)
_TASKS += BBH_TASKS


## AGI eval ##
class CustomAGIEvalEvaluationTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        prompt_function="agi_eval_prompt_no_letters",
        hf_repo="lighteval/agi_eval_en",
        hf_subset=None,
        #  metric=[Metrics.loglikelihood_acc_single_token],
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["train"],
        few_shots_split="validation",
        few_shots_select=None,
        suite=None,
        generation_size=-1,
        stop_sequence=None,
        output_regex=None,
        frozen=False,
    ):
        super().__init__(
            name=name,
            prompt_function=prompt_function,
            hf_repo=hf_repo,
            hf_subset=hf_subset,
            metric=metric,
            hf_avail_splits=hf_avail_splits,
            evaluation_splits=evaluation_splits,
            few_shots_split=few_shots_split,
            few_shots_select=few_shots_select,
            suite=suite,
            generation_size=generation_size,
            stop_sequence=stop_sequence,
            output_regex=output_regex,
            frozen=frozen,
        )


AGIEVAL_TASKS = [
    CustomAGIEvalEvaluationTask(name="agi_eval:aqua_rat", hf_subset="aqua_rat"),
    CustomAGIEvalEvaluationTask(name="agi_eval:logiqa-en", hf_subset="logiqa-en"),
    CustomAGIEvalEvaluationTask(name="agi_eval:lsat-ar", hf_subset="lsat-ar"),
    CustomAGIEvalEvaluationTask(name="agi_eval:lsat-lr", hf_subset="lsat-lr"),
    CustomAGIEvalEvaluationTask(name="agi_eval:lsat-rc", hf_subset="lsat-rc"),
    CustomAGIEvalEvaluationTask(
        name="agi_eval:math",
        hf_subset="math",
        prompt_function="agi_eval_math_prompt",
        metric=[Metrics.exact_match, Metrics.quasi_exact_match],
        generation_size=40,
    ),
    CustomAGIEvalEvaluationTask(name="agi_eval:sat-en", hf_subset="sat-en"),
    CustomAGIEvalEvaluationTask(name="agi_eval:sat-math", hf_subset="sat-math"),
]


def agi_eval_math_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[line["answer"]],
        gold_index=0,
        instruction="",
    )


def agi_eval_prompt(line, task_name: str = None):
    cleaned_options = [o.replace("(", "").replace(")", " ") for o in line["options"]]
    prompt = "The following are multiple choice questions (with answers).\n\n"
    prompt += line["question"] + "\n" + "\n".join(cleaned_options) + "\n"
    prompt += "Answer: "

    choices = LETTER_INDICES[: len(line["options"])]

    output = Doc(
        query=prompt,
        instruction="The following are multiple choice questions (with answers).\n\n",
        choices=None,  # updated below
        gold_index=None,  # updated below
    )

    if line["label"]:
        output.choices = choices
        output.gold_index = LETTER_INDICES.index(line["label"].strip())
    else:
        output.choices = [line["answer"]]
        output.gold_index = 0

    return output


def agi_eval_prompt_no_letters(line, task_name: str = None):
    cleaned_options = [
        " " + o.replace("(A)", "").replace("(B)", "").replace("(C)", "").replace("(D)", "").replace("(E)", "")
        for o in line["options"]
    ]

    output = Doc(
        query=line["question"],
        choices=cleaned_options,
        gold_index=LETTER_INDICES.index(line["label"].strip()),
        instruction="",
    )

    return output


# AGIEVAL_STRING = {t: f'custom|{t.name}|5|1' for t in AGIEVAL_TASKS}
AGIEVAL_STRING = [(t, f"custom|{t.name}|0|1") for t in AGIEVAL_TASKS]
_TASKS_STRINGS.extend(AGIEVAL_STRING)
_TASKS += AGIEVAL_TASKS


OPEN_LLM_LEADERBOARD_STRING = [
    "custom|arc:challenge|25|1",
    "custom|hellaswag|10|1",
    "lighteval|truthfulqa:mc|0|1",
    "custom|winogrande|5|1",
    "lighteval|gsm8k|5|1",
] + [f"custom|{t.name}|5|1" for t in MMLU_TASKS]


## HUMAN EVAL ##
#TODO @eliebak add human eval again
# human_eval = LightevalTaskConfig(
#         name="human_eval",
#         prompt_function="human_eval",
#         hf_repo="lighteval/human_eval",
#         metric=["human_eval_pass_at_1"],
#     ),


EARLY_SIGNAL_TASKS = ",".join([t[1] for t in COMMON_SENSE_REASONING_STRING] + [t[1] for t in MMLU_STRING])

# Convert to dict for lighteval
TASKS_TABLE = [asdict(task) for task in _TASKS]
# You can have a few pre-organised groups of tasks
# TODO @eliebak add math and code here 
TASKS_GROUPS = {
    "all": ",".join(t[1] for t in _TASKS_STRINGS),
    "early-signal": EARLY_SIGNAL_TASKS,
    "open-llm-leaderboard": ",".join(OPEN_LLM_LEADERBOARD_STRING),
}

if __name__ == "__main__":
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
