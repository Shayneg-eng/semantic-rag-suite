import os
import json
import numpy as np
import openai
import ollama
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================================================
# 1. CONFIGURATION & TEST BANK
# ==============================================================================

POE_API_KEY = os.getenv("POE_API_KEY", "")

# Ground Truth Test Bank (2 prompts per document)
TEST_SUITE = {
    "DOC_1.txt": [
        "How do I implement secure third-party access to my API using tokens?",
        "What are the best practices for managing access token expiration and refresh in OAuth2?"
    ],
    "DOC_2.txt": [
        "How do composite indexes improve database query performance?",
        "What is the difference between B-tree and hash indexes in SQL databases?"
    ],
    "DOC_3.txt": [
        "How do I decide between synchronous and asynchronous communication in microservices?",
        "What are the advantages of using a service registry for discovery in a distributed system?"
    ],
    "DOC_4.txt": [
        "What is the cache-aside pattern and how does it handle cache misses?",
        "How does Redis replication and sentinel ensure high availability during failure?"
    ],
    "DOC_5.txt": [
        "How can I reconstruct a request flow across distributed services using logs and correlation IDs?",
        "What are the differences between log levels like INFO, WARN, and ERROR in production?"
    ],
    "DOC_6.txt": [
        "What is the principle of least privilege and how does it apply to account security?",
        "How do parameterized queries protect against SQL injection attacks?"
    ],
    "DOC_7.txt": [
        "What are the differences between Layer 4 and Layer 7 load balancing?",
        "How does weighted round-robin distribution handle servers with different capacities?"
    ],
    "DOC_8.txt": [
        "What are the trade-offs between synchronous and asynchronous database replication?",
        "How do consensus algorithms like Raft manage automatic failover in distributed databases?"
    ],
    "DOC_9.txt": [
        "What is the difference between token bucket and leaky bucket rate limiting algorithms?",
        "How should clients implement backoff strategies when hitting HTTP 429 status codes?"
    ],
    "DOC_10.txt": [
        "How does the circuit breaker pattern protect services from cascading failures?",
        "What is exponential backoff with jitter and why is it used for network retries?"
    ]
}

SHIFT_PROMPTS = {
    "shift_1_entity": "Extract only the subject-verb-object relationships. Strip all adjectives and context. Format as simple declarative sentences.",
    "shift_2_abstraction": "Rewrite this using higher-level abstract concepts. (e.g., 'iPhone' -> 'mobile device', 'login' -> 'authentication').",
    "shift_3_certainty": "Remove all hedging (maybe, possibly, typically). Rewrite every sentence as an absolute, 100% certain fact.",
    "shift_5_passive": "Rewrite the entire text in the passive voice. Remove all mentions of who is doing the action.",
    "shift_6_negation": "Flip the polarity. Rewrite positive assertions as negative ones, and negative ones as positive.",
    "shift_7_density": "Summarize this into the shortest possible version that retains all unique information (maximum compression).",
    "shift_9_perspective": "Rewrite from the opposite stakeholder's perspective. If it's about a provider, rewrite as if from a consumer's view, and vice versa.",
    "shift_10_temporal": "Rewrite this with all temporal markers removed. Convert all tenses to present tense and remove time references (yesterday, next year, etc).",
    "shift_11_modality": "Rewrite replacing all modal verbs (could, should, might) with definitive statements. Convert all possibilities to actualities.",
    "shift_12_agent_flip": "Swap all active agents with their recipients or counterparts. If X does Y to Z, rewrite as Z does Y to X.",
    "shift_14_causality_invert": "Invert all cause-effect relationships. Rewrite as if effects are causes and causes are effects.",
    "shift_15_contrast": "Extract only contrasts, opposites, and comparative relationships. Emphasize what differs rather than similarities.",
    "shift_16_exemplification": "Convert all generic statements to specific examples, and all examples to generic patterns they represent."
}

poe_client = openai.OpenAI(api_key=POE_API_KEY, base_url="https://api.poe.com/v1")

# ==============================================================================
# 2. CORE ENGINE FUNCTIONS
# ==============================================================================

def get_embedding(text: str) -> np.ndarray:
    if not text or not text.strip():
        return np.zeros(768)
    response = ollama.embed(model='nomic-embed-text', input=text)
    return np.array(response['embeddings'][0])

def generate_shift(text: str, instruction: str) -> str:
    try:
        response = poe_client.chat.completions.create(
            model="llama-3.1-8b-cs",
            messages=[{"role": "user", "content": f"{instruction}\n\nTEXT:\n{text}"}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [WARNING] Shift generation failed: {e}")
        return text

def cosine_similarity(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return np.dot(v1, v2) / (n1 * n2) if n1 > 0 and n2 > 0 else 0

def _process_shift_parallel(args: Tuple) -> Tuple[str, np.ndarray]:
    """Process a single shift (for parallel execution)"""
    s_name, prompt, query_text, q_anchor_vec, noise_vec = args
    s_text = generate_shift(query_text, prompt)
    s_vec = get_embedding(s_text)
    delta = (s_vec - q_anchor_vec) - noise_vec
    return s_name, delta

# ==============================================================================
# 3. ENHANCED DIAGNOSTICS CLASS
# ==============================================================================

class DiagnosticsCollector:
    def __init__(self):
        self.query_results = []
        self.shift_performance = defaultdict(lambda: {
            'contribution': [],
            'correlation_with_success': [],
            'avg_alignment': []
        })
        self.per_doc_performance = defaultdict(lambda: {
            'rag_ranks': [],
            'fp_ranks': [],
            'rag_scores': [],
            'fp_scores': []
        })
        self.confusion_matrix = {
            'rag': defaultdict(int),
            'fingerprint': defaultdict(int)
        }
        self.score_distributions = {
            'rag': {'correct': [], 'incorrect': []},
            'fingerprint': {'correct': [], 'incorrect': []}
        }
        
    def record_query(self, query_data: Dict):
        """Record comprehensive data for each query"""
        self.query_results.append(query_data)
        
        # Track per-document performance
        target = query_data['target_doc']
        self.per_doc_performance[target]['rag_ranks'].append(query_data['rag_rank'])
        self.per_doc_performance[target]['fp_ranks'].append(query_data['fp_rank'])
        self.per_doc_performance[target]['rag_scores'].append(query_data['rag_top_score'])
        self.per_doc_performance[target]['fp_scores'].append(query_data['fp_top_score'])
        
        # Confusion matrix
        rag_pred = query_data['rag_top_doc']
        fp_pred = query_data['fp_top_doc']
        self.confusion_matrix['rag'][(target, rag_pred)] += 1
        self.confusion_matrix['fingerprint'][(target, fp_pred)] += 1
        
        # Score distributions
        self.score_distributions['rag']['correct' if rag_pred == target else 'incorrect'].append(
            query_data['rag_top_score']
        )
        self.score_distributions['fingerprint']['correct' if fp_pred == target else 'incorrect'].append(
            query_data['fp_top_score']
        )
        
        # Shift-level performance
        for shift_name, shift_data in query_data['shift_details'].items():
            self.shift_performance[shift_name]['avg_alignment'].append(shift_data['alignment'])
            self.shift_performance[shift_name]['contribution'].append(shift_data['contribution'])
            self.shift_performance[shift_name]['correlation_with_success'].append(
                1 if fp_pred == target else 0
            )
    
    def generate_report(self, output_dir: str = "diagnostics"):
        """Generate comprehensive diagnostic report"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Overall Performance Summary
        self._generate_summary_report(output_dir, timestamp)
        
        # 2. Per-Shift Analysis
        self._generate_shift_analysis(output_dir, timestamp)
        
        # 3. Per-Document Analysis
        self._generate_doc_analysis(output_dir, timestamp)
        
        # 4. Failure Case Analysis
        self._generate_failure_analysis(output_dir, timestamp)
        
        # 5. Score Distribution Analysis
        self._generate_score_distribution(output_dir, timestamp)
        
        # 6. Visualizations
        self._generate_visualizations(output_dir, timestamp)
        
        # 7. Detailed Query Log (JSON)
        self._save_query_log(output_dir, timestamp)
        
        print(f"\n✅ Diagnostics saved to: {output_dir}/")
        
    def _generate_summary_report(self, output_dir: str, timestamp: str):
        """Generate high-level summary"""
        total = len(self.query_results)
        rag_correct = sum(1 for q in self.query_results if q['rag_rank'] == 1)
        fp_correct = sum(1 for q in self.query_results if q['fp_rank'] == 1)
        
        rag_mrr = sum(1/q['rag_rank'] for q in self.query_results) / total
        fp_mrr = sum(1/q['fp_rank'] for q in self.query_results) / total
        
        rag_gaps = [q['rag_gap'] for q in self.query_results]
        fp_gaps = [q['fp_gap'] for q in self.query_results]
        
        report = f"""
{'='*80}
DIRECTIONAL RAG BENCHMARK REPORT
{'='*80}
Generated: {timestamp}
Total Queries: {total}

OVERALL PERFORMANCE
{'-'*80}
Metric                  | Standard RAG | Fingerprint V4 | Improvement
{'-'*80}
Top-1 Accuracy          | {rag_correct/total*100:6.2f}%     | {fp_correct/total*100:6.2f}%       | {(fp_correct-rag_correct)/total*100:+6.2f}%
Mean Reciprocal Rank    | {rag_mrr:6.4f}     | {fp_mrr:6.4f}       | {fp_mrr-rag_mrr:+6.4f}
Avg Confidence Gap      | {np.mean(rag_gaps):6.4f}     | {np.mean(fp_gaps):6.4f}       | {np.mean(fp_gaps)-np.mean(rag_gaps):+6.4f}
Median Gap              | {np.median(rag_gaps):6.4f}     | {np.median(fp_gaps):6.4f}       | {np.median(fp_gaps)-np.median(rag_gaps):+6.4f}
Std Dev Gap             | {np.std(rag_gaps):6.4f}     | {np.std(fp_gaps):6.4f}       | {np.std(fp_gaps)-np.std(rag_gaps):+6.4f}

RANK DISTRIBUTION
{'-'*80}
Rank | Standard RAG | Fingerprint V4
{'-'*80}
"""
        for rank in range(1, 6):
            rag_count = sum(1 for q in self.query_results if q['rag_rank'] == rank)
            fp_count = sum(1 for q in self.query_results if q['fp_rank'] == rank)
            report += f"  {rank}  |     {rag_count:2d}       |      {fp_count:2d}\n"
        
        # Win/Loss/Tie Analysis
        wins = sum(1 for q in self.query_results if q['fp_rank'] < q['rag_rank'])
        losses = sum(1 for q in self.query_results if q['fp_rank'] > q['rag_rank'])
        ties = sum(1 for q in self.query_results if q['fp_rank'] == q['rag_rank'])
        
        report += f"""
WIN/LOSS/TIE ANALYSIS
{'-'*80}
Fingerprint Wins:  {wins:2d} ({wins/total*100:.1f}%)
Standard RAG Wins: {losses:2d} ({losses/total*100:.1f}%)
Ties:              {ties:2d} ({ties/total*100:.1f}%)
"""
        
        with open(f"{output_dir}/summary_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        print(report)
    
    def _generate_shift_analysis(self, output_dir: str, timestamp: str):
        """Analyze individual shift contributions"""
        report = f"""
{'='*80}
PER-SHIFT PERFORMANCE ANALYSIS
{'='*80}

"""
        shift_stats = []
        
        for shift_name, data in self.shift_performance.items():
            avg_alignment = np.mean(data['avg_alignment'])
            avg_contribution = np.mean(data['contribution'])
            success_correlation = np.mean(data['correlation_with_success'])
            
            # Calculate variance and effectiveness
            alignment_std = np.std(data['avg_alignment'])
            contribution_std = np.std(data['contribution'])
            
            shift_stats.append({
                'name': shift_name,
                'avg_alignment': avg_alignment,
                'avg_contribution': avg_contribution,
                'success_correlation': success_correlation,
                'alignment_std': alignment_std,
                'contribution_std': contribution_std,
                'effectiveness_score': avg_contribution * success_correlation
            })
        
        # Sort by effectiveness
        shift_stats.sort(key=lambda x: x['effectiveness_score'], reverse=True)
        
        report += f"{'Shift':<25} | {'Avg Align':<10} | {'Avg Contrib':<12} | {'Success %':<10} | {'Effectiveness':<12}\n"
        report += f"{'-'*80}\n"
        
        for s in shift_stats:
            report += f"{s['name']:<25} | {s['avg_alignment']:>9.4f} | {s['avg_contribution']:>11.4f} | {s['success_correlation']*100:>9.1f}% | {s['effectiveness_score']:>11.4f}\n"
        
        report += f"\n\nSHIFT RECOMMENDATIONS\n{'-'*80}\n"
        
        # Identify top and bottom performers
        top_3 = shift_stats[:3]
        bottom_3 = shift_stats[-3:]
        
        report += "\n🟢 TOP PERFORMING SHIFTS (Keep & Optimize):\n"
        for s in top_3:
            report += f"  - {s['name']}: Effectiveness = {s['effectiveness_score']:.4f}\n"
            report += f"    → High contribution ({s['avg_contribution']:.4f}) with {s['success_correlation']*100:.1f}% correlation to success\n"
        
        report += "\n🔴 UNDERPERFORMING SHIFTS (Review or Remove):\n"
        for s in bottom_3:
            report += f"  - {s['name']}: Effectiveness = {s['effectiveness_score']:.4f}\n"
            if s['avg_contribution'] < 0.01:
                report += f"    ⚠️ Low contribution ({s['avg_contribution']:.4f}) - Consider removing\n"
            if s['success_correlation'] < 0.5:
                report += f"    ⚠️ Poor success correlation ({s['success_correlation']*100:.1f}%) - May add noise\n"
        
        with open(f"{output_dir}/shift_analysis_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        print(report)
    
    def _generate_doc_analysis(self, output_dir: str, timestamp: str):
        """Analyze per-document performance"""
        report = f"""
{'='*80}
PER-DOCUMENT PERFORMANCE ANALYSIS
{'='*80}

"""
        doc_stats = []
        
        for doc_name, data in self.per_doc_performance.items():
            rag_avg_rank = np.mean(data['rag_ranks'])
            fp_avg_rank = np.mean(data['fp_ranks'])
            rag_avg_score = np.mean(data['rag_scores'])
            fp_avg_score = np.mean(data['fp_scores'])
            
            improvement = rag_avg_rank - fp_avg_rank
            
            doc_stats.append({
                'name': doc_name,
                'rag_avg_rank': rag_avg_rank,
                'fp_avg_rank': fp_avg_rank,
                'rag_avg_score': rag_avg_score,
                'fp_avg_score': fp_avg_score,
                'rank_improvement': improvement,
                'score_improvement': fp_avg_score - rag_avg_score
            })
        
        # Sort by improvement
        doc_stats.sort(key=lambda x: x['rank_improvement'], reverse=True)
        
        report += f"{'Document':<15} | {'RAG Rank':<10} | {'FP Rank':<10} | {'Improvement':<12} | {'Score Δ':<10}\n"
        report += f"{'-'*80}\n"
        
        for d in doc_stats:
            improvement_symbol = "🟢" if d['rank_improvement'] > 0 else ("🔴" if d['rank_improvement'] < 0 else "⚪")
            report += f"{d['name']:<15} | {d['rag_avg_rank']:>9.2f} | {d['fp_avg_rank']:>9.2f} | {improvement_symbol} {d['rank_improvement']:>8.2f} | {d['score_improvement']:>+9.4f}\n"
        
        report += f"\n\nDOCUMENT-SPECIFIC INSIGHTS\n{'-'*80}\n"
        
        # Most improved
        most_improved = doc_stats[0]
        report += f"\n✅ Most Improved: {most_improved['name']}\n"
        report += f"   Rank: {most_improved['rag_avg_rank']:.2f} → {most_improved['fp_avg_rank']:.2f} (Δ {most_improved['rank_improvement']:+.2f})\n"
        
        # Least improved / regressed
        least_improved = doc_stats[-1]
        report += f"\n⚠️  Needs Attention: {least_improved['name']}\n"
        report += f"   Rank: {least_improved['rag_avg_rank']:.2f} → {least_improved['fp_avg_rank']:.2f} (Δ {least_improved['rank_improvement']:+.2f})\n"
        
        if least_improved['rank_improvement'] < 0:
            report += f"   → Fingerprint is WORSE than baseline - investigate shift quality for this doc type\n"
        
        with open(f"{output_dir}/doc_analysis_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        print(report)
    
    def _generate_failure_analysis(self, output_dir: str, timestamp: str):
        """Deep dive into failure cases"""
        report = f"""
{'='*80}
FAILURE CASE ANALYSIS
{'='*80}

"""
        # Cases where fingerprint failed but RAG succeeded
        fp_failures = [q for q in self.query_results if q['rag_rank'] == 1 and q['fp_rank'] > 1]
        
        # Cases where both failed
        both_failures = [q for q in self.query_results if q['rag_rank'] > 1 and q['fp_rank'] > 1]
        
        # Cases where fingerprint succeeded but RAG failed
        fp_wins = [q for q in self.query_results if q['fp_rank'] == 1 and q['rag_rank'] > 1]
        
        report += f"Total Queries: {len(self.query_results)}\n"
        report += f"Fingerprint Regressions: {len(fp_failures)} ({len(fp_failures)/len(self.query_results)*100:.1f}%)\n"
        report += f"Both Failed: {len(both_failures)} ({len(both_failures)/len(self.query_results)*100:.1f}%)\n"
        report += f"Fingerprint Rescued: {len(fp_wins)} ({len(fp_wins)/len(self.query_results)*100:.1f}%)\n\n"
        
        if fp_failures:
            report += f"\n{'='*80}\n"
            report += "FINGERPRINT REGRESSION CASES (RAG was correct, Fingerprint failed)\n"
            report += f"{'='*80}\n\n"
            
            for i, q in enumerate(fp_failures[:5], 1):  # Show top 5
                report += f"{i}. Query: {q['query'][:70]}...\n"
                report += f"   Target: {q['target_doc']} | RAG Predicted: {q['rag_top_doc']} | FP Predicted: {q['fp_top_doc']}\n"
                report += f"   RAG Score: {q['rag_top_score']:.4f} | FP Score: {q['fp_top_score']:.4f}\n"
                report += f"   FP Rank: {q['fp_rank']} | Gap: {q['fp_gap']:.4f}\n"
                
                # Analyze shift contributions
                report += "   Shift Contributions:\n"
                for shift_name, shift_data in sorted(q['shift_details'].items(), 
                                                     key=lambda x: x[1]['contribution'], reverse=True):
                    report += f"     - {shift_name}: {shift_data['contribution']:+.4f} (align: {shift_data['alignment']:.4f})\n"
                report += "\n"
        
        if fp_wins:
            report += f"\n{'='*80}\n"
            report += "FINGERPRINT RESCUE CASES (RAG failed, Fingerprint succeeded)\n"
            report += f"{'='*80}\n\n"
            
            for i, q in enumerate(fp_wins[:5], 1):
                report += f"{i}. Query: {q['query'][:70]}...\n"
                report += f"   Target: {q['target_doc']} | RAG Predicted: {q['rag_top_doc']} | FP Predicted: {q['fp_top_doc']}\n"
                report += f"   RAG Rank: {q['rag_rank']} | FP Rank: 1\n"
                report += f"   Key Winning Shifts:\n"
                for shift_name, shift_data in sorted(q['shift_details'].items(), 
                                                     key=lambda x: x[1]['contribution'], reverse=True)[:3]:
                    report += f"     - {shift_name}: {shift_data['contribution']:+.4f}\n"
                report += "\n"
        
        with open(f"{output_dir}/failure_analysis_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        print(report[:500] + "...\n(Full report saved to file)")
    
    def _generate_score_distribution(self, output_dir: str, timestamp: str):
        """Analyze score distributions"""
        report = f"""
{'='*80}
SCORE DISTRIBUTION ANALYSIS
{'='*80}

"""
        for method in ['rag', 'fingerprint']:
            method_name = "Standard RAG" if method == 'rag' else "Fingerprint V4"
            report += f"\n{method_name}\n{'-'*40}\n"
            
            correct_scores = self.score_distributions[method]['correct']
            incorrect_scores = self.score_distributions[method]['incorrect']
            
            if correct_scores:
                report += f"Correct Predictions:\n"
                report += f"  Mean:   {np.mean(correct_scores):.4f}\n"
                report += f"  Median: {np.median(correct_scores):.4f}\n"
                report += f"  Std:    {np.std(correct_scores):.4f}\n"
                report += f"  Min:    {np.min(correct_scores):.4f}\n"
                report += f"  Max:    {np.max(correct_scores):.4f}\n\n"
            
            if incorrect_scores:
                report += f"Incorrect Predictions:\n"
                report += f"  Mean:   {np.mean(incorrect_scores):.4f}\n"
                report += f"  Median: {np.median(incorrect_scores):.4f}\n"
                report += f"  Std:    {np.std(incorrect_scores):.4f}\n"
                report += f"  Min:    {np.min(incorrect_scores):.4f}\n"
                report += f"  Max:    {np.max(incorrect_scores):.4f}\n\n"
            
            if correct_scores and incorrect_scores:
                separation = np.mean(correct_scores) - np.mean(incorrect_scores)
                report += f"Score Separation (Correct - Incorrect): {separation:+.4f}\n"
                report += f"  → {'Good' if separation > 0.05 else 'Poor'} discrimination ability\n\n"
        
        with open(f"{output_dir}/score_distribution_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        print(report)
    
    def _generate_visualizations(self, output_dir: str, timestamp: str):
        """Generate diagnostic charts"""
        try:
            # 1. Rank Distribution Comparison
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Rank histogram
            rag_ranks = [q['rag_rank'] for q in self.query_results]
            fp_ranks = [q['fp_rank'] for q in self.query_results]
            
            axes[0, 0].hist([rag_ranks, fp_ranks], bins=range(1, 12), label=['Standard RAG', 'Fingerprint V4'], alpha=0.7)
            axes[0, 0].set_xlabel('Rank')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Rank Distribution Comparison')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Confidence gap comparison
            rag_gaps = [q['rag_gap'] for q in self.query_results]
            fp_gaps = [q['fp_gap'] for q in self.query_results]
            
            axes[0, 1].scatter(rag_gaps, fp_gaps, alpha=0.6)
            axes[0, 1].plot([0, max(max(rag_gaps), max(fp_gaps))], [0, max(max(rag_gaps), max(fp_gaps))], 'r--', alpha=0.5)
            axes[0, 1].set_xlabel('RAG Confidence Gap')
            axes[0, 1].set_ylabel('Fingerprint Confidence Gap')
            axes[0, 1].set_title('Confidence Gap Comparison')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Shift contribution analysis
            shift_names = list(self.shift_performance.keys())
            shift_contribs = [np.mean(self.shift_performance[s]['contribution']) for s in shift_names]
            shift_success = [np.mean(self.shift_performance[s]['correlation_with_success']) * 100 for s in shift_names]
            
            x_pos = np.arange(len(shift_names))
            axes[1, 0].bar(x_pos, shift_contribs, alpha=0.7)
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels([s.replace('shift_', '').replace('_', '\n') for s in shift_names], rotation=45, ha='right', fontsize=8)
            axes[1, 0].set_ylabel('Avg Contribution')
            axes[1, 0].set_title('Shift Contributions')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # Shift effectiveness (contribution × success rate)
            effectiveness = [c * s / 100 for c, s in zip(shift_contribs, shift_success)]
            axes[1, 1].bar(x_pos, effectiveness, alpha=0.7, color='green')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels([s.replace('shift_', '').replace('_', '\n') for s in shift_names], rotation=45, ha='right', fontsize=8)
            axes[1, 1].set_ylabel('Effectiveness Score')
            axes[1, 1].set_title('Shift Effectiveness (Contribution × Success Rate)')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/diagnostics_{timestamp}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # 2. Score distributions
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            for idx, (method, ax) in enumerate(zip(['rag', 'fingerprint'], axes)):
                correct = self.score_distributions[method]['correct']
                incorrect = self.score_distributions[method]['incorrect']
                
                if correct and incorrect:
                    ax.hist([correct, incorrect], bins=20, label=['Correct', 'Incorrect'], alpha=0.7)
                    ax.set_xlabel('Score')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f"{'Standard RAG' if method == 'rag' else 'Fingerprint V4'} Score Distribution")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/score_distributions_{timestamp}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Visualizations saved")
            
        except Exception as e:
            print(f"⚠️  Visualization generation failed: {e}")
    
    def _save_query_log(self, output_dir: str, timestamp: str):
        """Save detailed query-by-query JSON log"""
        with open(f"{output_dir}/query_log_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(self.query_results, f, indent=2)
        print(f"✅ Detailed query log saved")

# ==============================================================================
# 4. ENHANCED BENCHMARKING LOGIC
# ==============================================================================

def run_benchmark(index_file: str):
    with open(index_file, "r") as f:
        index = json.load(f)

    # Pre-calculate Global Noise Vectors
    print("\n[1/3] Calculating global noise vectors...")
    mean_deltas = {name: [] for name in SHIFT_PROMPTS.keys()}
    for doc_data in index.values():
        d_anchor = np.array(doc_data["anchor_vector"])
        for s_name, s_data in doc_data["shifts"].items():
            # Only process shifts that are in current SHIFT_PROMPTS
            if s_name in SHIFT_PROMPTS:
                mean_deltas[s_name].append(np.array(s_data["vector"]) - d_anchor)
    noise_vectors = {s_name: np.mean(deltas, axis=0) if deltas else np.zeros(768) for s_name, deltas in mean_deltas.items()}
    
    print("[2/3] Running benchmark queries...")
    diagnostics = DiagnosticsCollector()
    total_queries = 0

    for target_doc, queries in TEST_SUITE.items():
        # Normalize target_doc to match index keys (remove .txt extension)
        target_doc_normalized = target_doc.replace('.txt', '')
        
        for query in queries:
            total_queries += 1
            print(f"\n[Query {total_queries}/{sum(len(q) for q in TEST_SUITE.values())}] {target_doc}: '{query[:50]}...'")
            
            # --- PROCESS QUERY IN PARALLEL ---
            q_anchor_vec = get_embedding(query)
            q_deltas_clean = {}
            
            # Prepare shift processing tasks
            shift_tasks = [
                (s_name, prompt, query, q_anchor_vec, noise_vectors[s_name])
                for s_name, prompt in SHIFT_PROMPTS.items()
            ]
            
            # Process shifts in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(_process_shift_parallel, task): task[0] for task in shift_tasks}
                for future in as_completed(futures):
                    s_name, delta = future.result()
                    q_deltas_clean[s_name] = delta

            # --- SCORE ALL DOCS IN PARALLEL ---
            def score_document(doc_name_and_data):
                doc_name, doc_data = doc_name_and_data
                d_anchor = np.array(doc_data["anchor_vector"])
                
                # Standard RAG Score
                base_score = cosine_similarity(q_anchor_vec, d_anchor)
                
                # Fingerprint Score with detailed shift tracking
                alignments = []
                weights = []
                doc_shift_details = {}
                
                for s_name, q_delta_clean_val in q_deltas_clean.items():
                    if s_name in doc_data["shifts"]:
                        d_delta_clean = (np.array(doc_data["shifts"][s_name]["vector"]) - d_anchor) - noise_vectors[s_name]
                        alignment = cosine_similarity(q_delta_clean_val, d_delta_clean)
                        alignments.append(alignment)
                        
                        # Apply weights: 2.0x for "Big Three" (Temporal, Negation, Certainty)
                        if s_name in ["shift_10_temporal", "shift_6_negation", "shift_3_certainty"]:
                            weight = 2.0  # Big Three - proven high performers
                        elif s_name in ["shift_1_entity", "shift_9_perspective", "shift_12_agent_flip", 
                                        "shift_14_causality_invert", "shift_15_contrast"]:
                            weight = 1.5  # Secondary performers
                        else:
                            weight = 1.0
                        weights.append(weight)
                        
                        # Track contribution for TARGET document
                        if doc_name == target_doc_normalized:
                            doc_shift_details[s_name] = {
                                'alignment': alignment,
                                'contribution': alignment * base_score
                            }
                
                avg_alignment = np.average(alignments, weights=weights) if alignments else 0
                f_score = base_score * (1 + avg_alignment)
                
                return {
                    'doc_name': doc_name,
                    'rag_score': base_score,
                    'fp_score': f_score,
                    'shift_details': doc_shift_details
                }
            
            # Score all documents in parallel
            rag_results = []
            fingerprint_results = []
            shift_details = {}
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(score_document, item): item[0] for item in index.items()}
                for future in as_completed(futures):
                    result = future.result()
                    rag_results.append((result['doc_name'], result['rag_score']))
                    fingerprint_results.append((result['doc_name'], result['fp_score']))
                    if result['doc_name'] == target_doc_normalized:
                        shift_details = result['shift_details']

            # --- SORT RESULTS ---
            rag_results.sort(key=lambda x: x[1], reverse=True)
            fingerprint_results.sort(key=lambda x: x[1], reverse=True)
            
            # --- COLLECT DIAGNOSTICS ---
            rag_rank = next(i for i, (name, _) in enumerate(rag_results) if name == target_doc_normalized) + 1
            fp_rank = next(i for i, (name, _) in enumerate(fingerprint_results) if name == target_doc_normalized) + 1
            
            query_data = {
                'query': query,
                'target_doc': target_doc_normalized,
                'rag_rank': rag_rank,
                'fp_rank': fp_rank,
                'rag_top_doc': rag_results[0][0],
                'fp_top_doc': fingerprint_results[0][0],
                'rag_top_score': rag_results[0][1],
                'fp_top_score': fingerprint_results[0][1],
                'rag_gap': rag_results[0][1] - rag_results[1][1],
                'fp_gap': fingerprint_results[0][1] - fingerprint_results[1][1],
                'shift_details': shift_details,
                'all_rag_scores': {name: score for name, score in rag_results[:5]},
                'all_fp_scores': {name: score for name, score in fingerprint_results[:5]}
            }
            
            diagnostics.record_query(query_data)
            
            print(f"  RAG: Rank {rag_rank} | FP: Rank {fp_rank} | Gap: RAG={query_data['rag_gap']:.4f}, FP={query_data['fp_gap']:.4f}")

    # --- GENERATE COMPREHENSIVE DIAGNOSTICS ---
    print("\n[3/3] Generating diagnostic reports...")
    diagnostics.generate_report()

if __name__ == "__main__":
    if os.path.exists("rag_index.json"):
        run_benchmark("rag_index.json")
    else:
        print("❌ Please ensure rag_index.json is in the current directory.")