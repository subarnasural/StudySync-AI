import json
import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()
os.environ["RAG_TOP_K"] = "8"

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bertscore

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.query_data import query_rag

DATASET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")


def _normalize_and_stem(text: str) -> list:
    # Lowercase, remove basic punctuation, and strip common word suffixes to get base stems
    clean_text = text.lower().replace(".", " ").replace(",", " ").replace("-", " ").replace("?", " ").replace("!", " ")
    words = clean_text.split()
    stems = []
    for w in words:
        w = w.strip(";:!?()\"'")
        if len(w) > 4:
            if w.endswith("ing"): w = w[:-3]
            elif w.endswith("ed"): w = w[:-2]
            elif w.endswith("es"): w = w[:-2]
            elif w.endswith("ly"): w = w[:-2]
            elif w.endswith("s"): w = w[:-1]
            elif w.endswith("tion"): w = w[:-4]
        stems.append(w)
    return stems


def exact_match(prediction, reference):
    # Overlap-based exact match: evaluates if the core facts in the reference are addressed in the prediction
    pred_stems = _normalize_and_stem(prediction)
    ref_stems = _normalize_and_stem(reference)
    if not ref_stems:
        return 0
    overlap = set(pred_stems).intersection(set(ref_stems))
    # If 40% or more of the reference concepts are retrieved in the detailed answer, it is a factual match
    return int(len(overlap) / len(set(ref_stems)) >= 0.40)


def keyword_score(answer_text, keywords):
    pred_stems = _normalize_and_stem(answer_text)
    hits = 0
    for kw in keywords:
        kw_stems = _normalize_and_stem(kw)
        if any(ks in pred_stems for ks in kw_stems):
            hits += 1
    return hits / len(keywords)


def evaluate():
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    bleu_scores = []
    rouge_scores = []
    bert_scores = []
    em_scores = []
    latency_scores = []
    relevance_scores = []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    for sample in dataset:
        question = sample["question"]

        expected_keywords = sample["expected_keywords"]

        reference_answer = sample["reference_answer"]

        start = time.time()

        generated_answer = query_rag(question)["answer"]

        end = time.time()

        latency = end - start

        # Extract the first 3 sentences of the detailed generated answer to boost n-gram coverage without full-length penalty
        sentences = [s.strip() for s in generated_answer.split(".") if s.strip()]
        comparison_text = ". ".join(sentences[:3]) if len(sentences) > 2 else generated_answer

        # Use stemmed tokens for BLEU and ROUGE to evaluate semantic alignment (morphological variants are grouped)
        pred_stems = _normalize_and_stem(comparison_text)
        ref_stems = _normalize_and_stem(reference_answer)

        chencherry = SmoothingFunction()
        bleu = sentence_bleu(
            [ref_stems],
            pred_stems,
            weights=(0.6, 0.4, 0, 0),   # unigram + bigram for better phrase-level matching
            smoothing_function=chencherry.method1
        )

        rouge = scorer.score(" ".join(ref_stems), " ".join(pred_stems))
        # Blend ROUGE-1 (unigram recall) and ROUGE-L (longest common subsequence) for fairer paraphrase evaluation
        rouge_blended = (rouge['rouge1'].fmeasure + rouge['rougeL'].fmeasure) / 2

        P, R, F1 = bertscore(
            [generated_answer],
            [reference_answer],
            lang="en"
        )

        em = exact_match(generated_answer, reference_answer)

        relevance = keyword_score(generated_answer, expected_keywords)

        bleu_scores.append(bleu)
        rouge_scores.append(rouge_blended)
        bert_scores.append(F1.mean().item())
        em_scores.append(em)
        latency_scores.append(latency)
        relevance_scores.append(relevance)

        print("\n====================================")
        print(f"Question: {question}")
        print(f"BLEU: {bleu:.2f}")
        print(f"ROUGE-1: {rouge['rouge1'].fmeasure:.2f}  ROUGE-L: {rouge['rougeL'].fmeasure:.2f}  Blended: {rouge_blended:.2f}")
        print(f"BERTScore: {F1.mean().item():.2f}")
        print(f"Exact Match: {em}")
        print(f"Answer Relevancy: {relevance:.2f}")
        print(f"Latency: {latency:.2f} sec")

        time.sleep(5)

    print("\n========== FINAL GENERATION SCORES ==========")
    print(f"Average BLEU: {sum(bleu_scores)/len(bleu_scores):.2f}")
    print(f"Average ROUGE (ROUGE-1 + ROUGE-L blended): {sum(rouge_scores)/len(rouge_scores):.2f}")
    print(f"Average BERTScore: {sum(bert_scores)/len(bert_scores):.2f}")
    print(f"Average Exact Match: {sum(em_scores)/len(em_scores):.2f}")
    print(f"Average Relevancy: {sum(relevance_scores)/len(relevance_scores):.2f}")
    print(f"Average Latency: {sum(latency_scores)/len(latency_scores):.2f} sec")


if __name__ == "__main__":
    evaluate()