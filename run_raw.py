import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns

def train_raw_model():
    """
    Train command:
    python run.py --model google/electra-small-discriminator --task nli --dataset snli \
    --do_train --do_eval --output_dir ./baseline_model --num_train_epochs 3 \
    --per_device_train_batch_size 32
    """
    # Modify run.py to use trainer.train() instead of train_model_cartography
    # Change this line in run.py:
    # if training_args.do_train:
    #     trainer.train()
    pass

def analyze_model_performance(model_path="./baseline_model", output_file="model_analysis.json"):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Load evaluation predictions
    with open(f"{model_path}/eval_predictions.jsonl", "r") as f:
        predictions = [json.loads(line) for line in f]

    # Initialize analysis metrics
    analysis = {
        "overall_accuracy": 0,
        "per_class_accuracy": defaultdict(float),
        "per_class_count": defaultdict(int),
        "error_patterns": defaultdict(int),
        "length_based_accuracy": defaultdict(list),
        "confidence_analysis": defaultdict(list)
    }

    # Analyze predictions
    total = len(predictions)
    correct = 0
    true_labels = []
    pred_labels = []

    for pred in predictions:
        true_label = pred["label"]
        pred_label = pred["predicted_label"]
        true_labels.append(true_label)
        pred_labels.append(pred_label)

        # Overall accuracy
        if true_label == pred_label:
            correct += 1
            analysis["per_class_accuracy"][true_label] += 1
        
        # Error patterns
        if true_label != pred_label:
            error_key = f"{true_label}->{pred_label}"
            analysis["error_patterns"][error_key] += 1

        # Per class counts
        analysis["per_class_count"][true_label] += 1

        # Length-based analysis
        premise_length = len(pred["premise"].split())
        hypothesis_length = len(pred["hypothesis"].split())
        analysis["length_based_accuracy"][premise_length].append(true_label == pred_label)

        # Confidence analysis
        confidence = max(pred["predicted_scores"])
        analysis["confidence_analysis"]["confidence"].append(confidence)
        analysis["confidence_analysis"]["correct"].append(true_label == pred_label)

    # Calculate final metrics
    analysis["overall_accuracy"] = correct / total
    for label in analysis["per_class_accuracy"]:
        analysis["per_class_accuracy"][label] /= analysis["per_class_count"][label]

    # Generate confusion matrix
    labels = sorted(list(set(true_labels)))
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f"{model_path}/confusion_matrix.png")
    plt.close()

    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    correct_conf = [c for c, corr in zip(analysis["confidence_analysis"]["confidence"], 
                                       analysis["confidence_analysis"]["correct"]) if corr]
    wrong_conf = [c for c, corr in zip(analysis["confidence_analysis"]["confidence"], 
                                      analysis["confidence_analysis"]["correct"]) if not corr]
    plt.hist([correct_conf, wrong_conf], label=['Correct', 'Wrong'], bins=30, alpha=0.7)
    plt.title('Confidence Distribution for Correct vs Wrong Predictions')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(f"{model_path}/confidence_distribution.png")
    plt.close()

    # Save analysis results
    with open(f"{model_path}/{output_file}", "w") as f:
        json.dump(analysis, f, indent=2)

    return analysis

def print_analysis_summary(analysis):
    print("\nModel Performance Analysis Summary")
    print("="*40)
    print(f"Overall Accuracy: {analysis['overall_accuracy']:.3f}")
    print("\nPer-Class Accuracy:")
    for label, acc in analysis['per_class_accuracy'].items():
        print(f"  {label}: {acc:.3f}")
    print("\nCommon Error Patterns:")
    sorted_errors = sorted(analysis['error_patterns'].items(), key=lambda x: x[1], reverse=True)
    for error_type, count in sorted_errors[:5]:
        print(f"  {error_type}: {count}")

def analyze_challenging_examples(model_path="./baseline_model"):
    """Analyze specific challenging examples and patterns"""
    with open(f"{model_path}/incorrect_predictions.jsonl", "r") as f:
        errors = json.load(f)

    # Analyze error patterns
    error_categories = defaultdict(list)
    for error in errors:
        # Negation analysis
        if any(neg in error["premise"].lower() for neg in ["not", "no", "never"]):
            error_categories["negation"].append(error)
        
        # Length mismatch
        premise_len = len(error["premise"].split())
        hypothesis_len = len(error["hypothesis"].split())
        if abs(premise_len - hypothesis_len) > 10:
            error_categories["length_mismatch"].append(error)
        
        # High confidence errors
        if max(error["predicted_scores"]) > 0.9:
            error_categories["high_confidence_errors"].append(error)

    # Print analysis
    print("\nChallenging Examples Analysis")
    print("="*40)
    for category, examples in error_categories.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        print(f"Total count: {len(examples)}")
        if examples:
            # Print a representative example
            example = examples[0]
            print("\nExample:")
            print(f"Premise: {example['premise']}")
            print(f"Hypothesis: {example['hypothesis']}")
            print(f"True Label: {example['label']}")
            print(f"Predicted Label: {example['predicted_label']}")

if __name__ == "__main__":
    # First train the raw model by running the modified run.py
    # train_raw_model()
    
    # Then analyze the model's performance
    analysis = analyze_model_performance()
    print_analysis_summary(analysis)
    analyze_challenging_examples()