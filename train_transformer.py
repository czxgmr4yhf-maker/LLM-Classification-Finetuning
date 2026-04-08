import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

LABEL_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]
ID_COLUMN = "id"


# 将任意输入规整成安全的普通字符串。
# 这里主要是为了防止原始数据里混入异常字符，导致后续 json / tokenizer / pandas 出错。
def normalize_text(value: str) -> str:
    return str(value).encode("utf-8", "replace").decode("utf-8").strip()


# 原始 csv 中的 prompt / response 字段长得像 ["...", "..."] 这样的列表字符串。
# 这个函数把它解析成普通文本，并用换行把多轮内容连接起来。
def parse_text_list(value: str) -> str:
    text = normalize_text(value)
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        return text
    return "\n".join(normalize_text(item) for item in items if normalize_text(item))


# 把一条样本整理成单段输入文本。
# 这样 Transformer 在同一个输入里就能同时看到题目、回答A、回答B，并学习它们之间的关系。
def build_model_input(prompt: str, response_a: str, response_b: str) -> str:
    return (
        "Instruction:\n"
        f"{prompt}\n\n"
        "Response A:\n"
        f"{response_a}\n\n"
        "Response B:\n"
        f"{response_b}"
    )


# 读取 csv 后完成三件事：
# 1. 把列表字符串解析成普通文本
# 2. 把 prompt / response_a / response_b 拼成 model_input
# 3. 如果是训练集，再把三列 one-hot 标签压成单列类别编号 label
def load_dataframe(csv_path: str, is_train: bool) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError as exc:
        try:
            df = pd.read_csv(csv_path, engine="python")
        except pd.errors.ParserError as fallback_exc:
            raise RuntimeError(
                f"Failed to parse CSV file: {csv_path}. "
                "The file is likely truncated or contains an unclosed quote. "
                "Please re-download or re-copy the CSV and verify it opens normally. "
                f"Original error: {exc}. Fallback error: {fallback_exc}"
            ) from fallback_exc
    df["prompt_text"] = df["prompt"].apply(parse_text_list)
    df["response_a_text"] = df["response_a"].apply(parse_text_list)
    df["response_b_text"] = df["response_b"].apply(parse_text_list)
    df["model_input"] = df.apply(
        lambda row: build_model_input(
            row["prompt_text"], row["response_a_text"], row["response_b_text"]
        ),
        axis=1,
    )
    if is_train:
        df["label"] = df[LABEL_COLUMNS].values.argmax(axis=1)
    return df


# 把模型输出的 logits 转成概率分布。
# logits 是原始分数，可能有负数，也不会自动加起来等于 1；
# softmax 之后才是三分类概率，可用于 accuracy / log_loss / 提交文件。
def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim == 1:
        logits = logits[None, :]
    logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-15, None)
    probs = np.nan_to_num(probs, nan=1.0 / len(LABEL_COLUMNS), posinf=1.0, neginf=0.0)
    return probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-15, None)


# 在验证集上计算两个指标：
# 1. accuracy：看分类对了多少
# 2. log_loss：看模型给真实类别分配了多大概率
# 这里用 eval_pred.predictions / eval_pred.label_ids，而不是直接解包，
# 是为了兼容不同 transformers 版本返回的 EvalPrediction 对象。
def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]

    probs = softmax(logits)
    preds = probs.argmax(axis=1)
    accuracy = float((preds == labels).mean())

    label_one_hot = np.eye(len(LABEL_COLUMNS))[labels]
    log_loss = float(-(label_one_hot * np.log(probs + 1e-15)).sum(axis=1).mean())
    return {"accuracy": accuracy, "log_loss": log_loss}


# 用 tokenizer 把文本编码成模型可用的 input_ids / attention_mask。
# batched=True 表示一次处理一批样本，会比一条一条处理更高效。
# tokenization 完成后会删掉不再需要的原始文本列，减少内存占用。
def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    removable_columns = [
        col for col in dataset.column_names if col not in {"label", ID_COLUMN, "model_input"}
    ]
    tokenized = dataset.map(
        lambda batch: tokenizer(
            batch["model_input"],
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
        remove_columns=removable_columns,
    )
    return tokenized.remove_columns([col for col in ["model_input"] if col in tokenized.column_names])


# 构造 Hugging Face 的 TrainingArguments。
# 这个对象只负责“训练怎么进行”，例如学习率、batch size、保存策略、评估频率等。
# 这里额外做了接口兼容，因为不同版本 transformers 对参数名有细微变化。
def build_training_arguments(
    output_dir: str,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    num_epochs: int,
    weight_decay: float,
    force_cpu: bool,
):
    args_kwargs = {
        "output_dir": output_dir,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_log_loss",
        "greater_is_better": False,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "num_train_epochs": num_epochs,
        "weight_decay": weight_decay,
        "logging_steps": 50,
        "save_total_limit": 2,
        "fp16": torch.cuda.is_available() and not force_cpu,
        "report_to": "none",
    }
    training_arg_names = inspect.signature(TrainingArguments.__init__).parameters
    if force_cpu and "use_cpu" in training_arg_names:
        args_kwargs["use_cpu"] = True
    if "evaluation_strategy" in training_arg_names:
        args_kwargs["evaluation_strategy"] = "epoch"
    else:
        args_kwargs["eval_strategy"] = "epoch"
    return TrainingArguments(**args_kwargs)


# 构造 Trainer 对象。
# 这个对象负责把模型、数据集、训练参数、padding 规则、评估函数真正组装起来。
# 后面调用 trainer.train() / trainer.predict() 都是基于它完成的。
def build_trainer(
    model,
    args,
    data_collator,
    tokenizer,
    train_dataset=None,
    eval_dataset=None,
    compute_metrics_fn=None,
):
    trainer_kwargs = {
        "model": model,
        "args": args,
        "data_collator": data_collator,
    }
    if train_dataset is not None:
        trainer_kwargs["train_dataset"] = train_dataset
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    if compute_metrics_fn is not None:
        trainer_kwargs["compute_metrics"] = compute_metrics_fn

    trainer_arg_names = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_arg_names:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        trainer_kwargs["processing_class"] = tokenizer

    return Trainer(**trainer_kwargs)


# make_trainer 是“总装函数”：
# 1. 加载 tokenizer
# 2. 加载序列分类模型
# 3. 对训练集和验证集做 tokenization
# 4. 创建 TrainingArguments
# 5. 创建最终可训练的 Trainer
def make_trainer(
    model_name: str,
    output_dir: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    max_length: int,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    num_epochs: int,
    weight_decay: float,
    force_cpu: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_COLUMNS),
    )

    tokenized_train = tokenize_dataset(train_dataset, tokenizer, max_length=max_length)
    tokenized_eval = tokenize_dataset(eval_dataset, tokenizer, max_length=max_length)

    args = build_training_arguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
        force_cpu=force_cpu,
    )

    trainer = build_trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics_fn=compute_metrics,
    )
    return trainer, tokenizer


# 对单个 fold 的验证集做预测，并保存详细结果：
# - 每条样本的 id
# - 真实标签
# - 预测标签
# - 三个类别的预测概率
# 同时返回这一折的 accuracy / log_loss，便于最后做 K 折平均。
def save_validation_predictions(trainer: Trainer, output_dir: str) -> dict:
    eval_dataset = trainer.eval_dataset
    pred_output = trainer.predict(eval_dataset)
    logits = pred_output.predictions[0] if isinstance(pred_output.predictions, tuple) else pred_output.predictions
    probs = softmax(logits)

    valid_ids = np.array(eval_dataset[ID_COLUMN])
    valid_labels = np.array(eval_dataset["label"])
    preds = pd.DataFrame(
        {
            ID_COLUMN: valid_ids,
            "true_label": valid_labels,
            "pred_label": probs.argmax(axis=1),
            LABEL_COLUMNS[0]: probs[:, 0],
            LABEL_COLUMNS[1]: probs[:, 1],
            LABEL_COLUMNS[2]: probs[:, 2],
        }
    )
    preds.to_csv(Path(output_dir) / "validation_predictions.csv", index=False)

    metrics = compute_metrics(type("EvalPred", (), {"predictions": logits, "label_ids": valid_labels})())
    return metrics


# 把多个 fold 训练出的模型参数做逐项平均，生成一个新的“平均模型”。
# 这里的思路就是你要求的方案：5 次训练 -> 5 套参数 -> 对同名参数求平均。
# 注意，这里平均的是模型权重，不是预测概率。
def average_model_weights(model_dirs: list[str], model_name: str, output_dir: str) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    averaged_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_COLUMNS),
    )
    averaged_state = averaged_model.state_dict()
    summed_state = {name: torch.zeros_like(param) for name, param in averaged_state.items()}

    for model_dir in model_dirs:
        fold_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        fold_state = fold_model.state_dict()
        for name in summed_state:
            summed_state[name] += fold_state[name].to(summed_state[name].dtype)

    num_models = float(len(model_dirs))
    for name in summed_state:
        averaged_state[name] = summed_state[name] / num_models

    averaged_model.load_state_dict(averaged_state)
    averaged_model.save_pretrained(output_path)
    AutoTokenizer.from_pretrained(model_dirs[0], use_fast=False).save_pretrained(output_path)
    return str(output_path)


# 这是完整的 K 折训练主流程。
# 它会：
# 1. 读入训练集
# 2. 用 StratifiedKFold 保持类别分布一致地拆成 K 折
# 3. 每一折训练一个模型
# 4. 保存每一折的验证预测和指标
# 5. 对 K 个模型的权重做平均，生成最终模型
def train_kfold(
    train_path: str,
    output_dir: str,
    model_name: str,
    n_splits: int,
    max_length: int,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    num_epochs: int,
    weight_decay: float,
    seed: int,
    max_train_samples: int | None,
    force_cpu: bool,
):
    df = load_dataframe(train_path, is_train=True)
    if max_train_samples:
        df = df.sample(n=max_train_samples, random_state=seed).reset_index(drop=True)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []
    fold_model_dirs = []

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(df, df["label"]), start=1):
        # 每一折都重新切出当前的训练子集和验证子集。
        fold_dir = output_root / f"fold_{fold_idx}"
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)

        train_dataset = Dataset.from_pandas(
            train_df[[ID_COLUMN, "model_input", "label"]], preserve_index=False
        )
        valid_dataset = Dataset.from_pandas(
            valid_df[[ID_COLUMN, "model_input", "label"]], preserve_index=False
        )

        trainer, tokenizer = make_trainer(
            model_name=model_name,
            output_dir=str(fold_dir),
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            max_length=max_length,
            learning_rate=learning_rate,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_epochs=num_epochs,
            weight_decay=weight_decay,
            force_cpu=force_cpu,
        )

        trainer.train()
        trainer.save_model(str(fold_dir))
        tokenizer.save_pretrained(str(fold_dir))
        metrics = save_validation_predictions(trainer, str(fold_dir))
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)
        fold_model_dirs.append(str(fold_dir))
        print(f"fold {fold_idx} metrics: {metrics}")

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(output_root / "fold_metrics.csv", index=False)

    mean_metrics = {
        "accuracy": float(metrics_df["accuracy"].mean()),
        "log_loss": float(metrics_df["log_loss"].mean()),
    }
    print("mean_cv_metrics:", mean_metrics)

    # 用所有 fold 的最终权重做平均，得到一个单独可加载的 averaged_model。
    averaged_model_dir = average_model_weights(
        model_dirs=fold_model_dirs,
        model_name=model_name,
        output_dir=str(output_root / "averaged_model"),
    )
    return {
        "fold_metrics": metrics_df,
        "mean_metrics": mean_metrics,
        "averaged_model_dir": averaged_model_dir,
    }


# 用“平均后的模型”对测试集做最终预测，生成提交文件。
# 这里不会再训练，只做推理。
def predict(
    test_path: str,
    model_dir: str,
    submission_path: str,
    max_length: int,
    eval_batch_size: int,
    max_test_samples: int | None,
):
    test_df = load_dataframe(test_path, is_train=False)
    if max_test_samples:
        test_df = test_df.head(max_test_samples).copy()

    dataset = Dataset.from_pandas(test_df[[ID_COLUMN, "model_input"]], preserve_index=False)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenized = tokenize_dataset(dataset, tokenizer, max_length=max_length)

    trainer = build_trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(Path(model_dir) / "predict_tmp"),
            per_device_eval_batch_size=eval_batch_size,
            report_to="none",
        ),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    pred_output = trainer.predict(tokenized)
    logits = pred_output.predictions[0] if isinstance(pred_output.predictions, tuple) else pred_output.predictions
    probs = softmax(logits)

    submission = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN].values,
            LABEL_COLUMNS[0]: probs[:, 0],
            LABEL_COLUMNS[1]: probs[:, 1],
            LABEL_COLUMNS[2]: probs[:, 2],
        }
    )
    submission.to_csv(submission_path, index=False)
    print(f"submission saved to {submission_path}")
    return submission


# 你以后主要只需要改这里的 config。
# 这一个字典集中控制了训练路径、模型名称、K 折数量、学习率、batch size、epoch 等全部关键参数。
def main():
    config = {
        "train_path": "train.csv",                  # 训练集路径
        "test_path": "test.csv",                    # 测试集路径
        "output_dir": "outputs/deberta_kfold",      # 所有fold模型和平均模型的保存目录
        "submission_path": "submission.csv",        # 最终提交文件路径
        "model_name": "microsoft/deberta-v3-small", # 预训练模型名称
        "n_splits": 5,                               # K折数量
        "max_length": 512,                           # 最大token长度
        "learning_rate": 2e-5,                       # 学习率
        "train_batch_size": 4,                       # 训练batch size
        "eval_batch_size": 8,                        # 验证/预测batch size
        "num_epochs": 2,                             # 每一折训练轮数
        "weight_decay": 0.01,                        # 权重衰减
        "seed": 42,                                  # 随机种子
        "max_train_samples": None,                   # 只取前面部分训练样本时填写整数，否则填None
        "max_test_samples": None,                    # 只取前面部分测试样本时填写整数，否则填None
        "force_cpu": False,                          # Mac上不稳定时可改成True
    }

    training_result = train_kfold(
        train_path=config["train_path"],
        output_dir=config["output_dir"],
        model_name=config["model_name"],
        n_splits=config["n_splits"],
        max_length=config["max_length"],
        learning_rate=config["learning_rate"],
        train_batch_size=config["train_batch_size"],
        eval_batch_size=config["eval_batch_size"],
        num_epochs=config["num_epochs"],
        weight_decay=config["weight_decay"],
        seed=config["seed"],
        max_train_samples=config["max_train_samples"],
        force_cpu=config["force_cpu"],
    )

    # 训练完成后，直接用平均模型做测试集预测。
    predict(
        test_path=config["test_path"],
        model_dir=training_result["averaged_model_dir"],
        submission_path=config["submission_path"],
        max_length=config["max_length"],
        eval_batch_size=config["eval_batch_size"],
        max_test_samples=config["max_test_samples"],
    )


if __name__ == "__main__":
    main()
