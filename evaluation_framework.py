import json
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class EvaluationResult:

    question: str
    answer: str
    source_documents: List[Dict]
    effectiveness_score: float
    faithfulness_score: float
    source_attribution_score: float
    overall_score: float
    timestamp: str

class RAGEvaluator:

    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.evaluation_results = []
        

        self.ground_truth_data = self._load_ground_truth()
    
    def _load_ground_truth(self) -> List[Dict]:

        return [
            {
                "question": "What are the crime statistics for Greater Sydney?",
                "expected_keywords": ["Greater Sydney", "crime", "statistics", "rate", "count", "per 100000"],
                "expected_sources": ["Greater Sydney"]
            },
            {
                "question": "Compare crime rates between Greater Sydney and NSW Regional areas",
                "expected_keywords": ["compare", "Greater Sydney", "NSW Regional", "rate", "per 100000", "difference"],
                "expected_sources": ["Greater Sydney", "NSW Regional"]
            },
            {
                "question": "Which region has the highest assault rates?",
                "expected_keywords": ["highest", "assault", "rate", "region", "Greater Sydney", "NSW Regional", "New South Wales"],
                "expected_sources": ["Greater Sydney", "NSW Regional", "New South Wales"]
            },
            {
                "question": "What are the most common types of crimes in NSW?",
                "expected_keywords": ["common", "types", "crimes", "NSW", "Assault", "Theft", "Drug", "Break and enter"],
                "expected_sources": ["New South Wales"]
            },
            {
                "question": "How has domestic violence related assault changed over time?",
                "expected_keywords": ["domestic violence", "assault", "changed", "time", "trend", "increasing", "decreasing"],
                "expected_sources": ["Greater Sydney", "NSW Regional", "New South Wales"]
            },
            {
                "question": "What crime trends have occurred from 2015 to 2024?",
                "expected_keywords": ["trends", "2015", "2024", "crime", "increasing", "decreasing", "stable"],
                "expected_sources": ["Greater Sydney", "NSW Regional", "New South Wales"]
            },
            {
                "question": "What are the murder rates across NSW regions?",
                "expected_keywords": ["murder", "rate", "regions", "Greater Sydney", "NSW Regional", "New South Wales"],
                "expected_sources": ["Greater Sydney", "NSW Regional", "New South Wales"]
            },
            {
                "question": "How prevalent is motor vehicle theft in Greater Sydney?",
                "expected_keywords": ["motor vehicle theft", "prevalent", "Greater Sydney", "rate", "count"],
                "expected_sources": ["Greater Sydney"]
            },
            {
                "question": "Which areas have the highest drug-related crime rates?",
                "expected_keywords": ["drug", "highest", "rate", "areas", "dealing", "trafficking", "possession"],
                "expected_sources": ["Greater Sydney", "NSW Regional", "New South Wales"]
            },
            {
                "question": "Show me all drug-related offenses in NSW",
                "expected_keywords": ["drug", "offenses", "NSW", "dealing", "trafficking", "possession", "cannabis", "amphetamines"],
                "expected_sources": ["New South Wales"]
            }
        ]
    
    def evaluate_effectiveness(self, question: str, answer: str, expected_keywords: List[str]) -> float:

        if not answer or not expected_keywords:
            return 0.0
        
        answer_lower = answer.lower()
        keyword_matches = 0
        
        for keyword in expected_keywords:
            if keyword.lower() in answer_lower:
                keyword_matches += 1
        
        return keyword_matches / len(expected_keywords) if expected_keywords else 0.0
    
    def evaluate_faithfulness(self, answer: str, source_documents: List[Dict]) -> float:

        if not answer or not source_documents:
            return 0.0
        
        answer_facts = self._extract_facts(answer)
        
        if not answer_facts:
            return 0.0
        
        supported_facts = 0
        
        for fact in answer_facts:
            for doc in source_documents:
                if fact.lower() in doc['content'].lower():
                    supported_facts += 1
                    break
        
        return supported_facts / len(answer_facts) if answer_facts else 0.0
    
    def evaluate_source_attribution(self, source_documents: List[Dict], expected_sources: List[str]) -> float:

        if not source_documents:
            return 0.0
        
        if not expected_sources:
            return 1.0 if source_documents else 0.0
        
        actual_sources = set()
        for doc in source_documents:
            if 'lga' in doc['metadata']:
                actual_sources.add(doc['metadata']['lga'])
        
        expected_sources_set = set(expected_sources)
        
        intersection = actual_sources.intersection(expected_sources_set)
        
        if not expected_sources_set:
            return 1.0
        
        return len(intersection) / len(expected_sources_set)
    
    def _extract_facts(self, text: str) -> List[str]:
        facts = []
        
        import re
        numbers = re.findall(r'\d+', text)
        facts.extend(numbers)
        
        crime_types = ['Murder', 'Assault', 'Robbery', 'Break and enter', 'Motor vehicle theft', 'Drug', 'Sexual assault', 'Fraud', 'Domestic violence']
        for crime_type in crime_types:
            if crime_type in text:
                facts.append(crime_type)
        
        lgas = ['Greater Sydney', 'NSW Regional', 'New South Wales']
        for lga in lgas:
            if lga in text:
                facts.append(lga)
        
        trends = ['increasing', 'decreasing', 'stable']
        for trend in trends:
            if trend in text.lower():
                facts.append(trend)
        
        return facts
    
    def run_evaluation(self) -> List[EvaluationResult]:
        print("Starting RAG system evaluation...")
        
        for gt_data in self.ground_truth_data:
            print(f"Evaluating: {gt_data['question']}")
            
            try:
                result = self.pipeline.query(gt_data['question'])
                
                effectiveness = self.evaluate_effectiveness(
                    gt_data['question'],
                    result['answer'],
                    gt_data['expected_keywords']
                )
                
                faithfulness = self.evaluate_faithfulness(
                    result['answer'],
                    result['source_documents']
                )
                
                source_attribution = self.evaluate_source_attribution(
                    result['source_documents'],
                    gt_data['expected_sources']
                )
                
                overall_score = (effectiveness * 0.4 + faithfulness * 0.4 + source_attribution * 0.2)
                
                eval_result = EvaluationResult(
                    question=gt_data['question'],
                    answer=result['answer'],
                    source_documents=result['source_documents'],
                    effectiveness_score=effectiveness,
                    faithfulness_score=faithfulness,
                    source_attribution_score=source_attribution,
                    overall_score=overall_score,
                    timestamp=datetime.now().isoformat()
                )
                
                self.evaluation_results.append(eval_result)
                
                print(f"  Effectiveness: {effectiveness:.2f}")
                print(f"  Faithfulness: {faithfulness:.2f}")
                print(f"  Source Attribution: {source_attribution:.2f}")
                print(f"  Overall Score: {overall_score:.2f}")
                
            except Exception as e:
                print(f"Error evaluating question '{gt_data['question']}': {str(e)}")
                continue
        
        return self.evaluation_results
    
    def generate_report(self) -> Dict:
        if not self.evaluation_results:
            return {"error": "No evaluation results available"}
        
    
        total_questions = len(self.evaluation_results)
        avg_effectiveness = sum(r.effectiveness_score for r in self.evaluation_results) / total_questions
        avg_faithfulness = sum(r.faithfulness_score for r in self.evaluation_results) / total_questions
        avg_source_attribution = sum(r.source_attribution_score for r in self.evaluation_results) / total_questions
        avg_overall = sum(r.overall_score for r in self.evaluation_results) / total_questions
        
        unanswered_questions = sum(1 for r in self.evaluation_results if r.overall_score < 0.3)
        unanswered_percentage = (unanswered_questions / total_questions) * 100
        
        report = {
            "evaluation_summary": {
                "total_questions": total_questions,
                "average_effectiveness": round(avg_effectiveness, 3),
                "average_faithfulness": round(avg_faithfulness, 3),
                "average_source_attribution": round(avg_source_attribution, 3),
                "average_overall_score": round(avg_overall, 3),
                "unanswered_questions": unanswered_questions,
                "unanswered_percentage": round(unanswered_percentage, 1)
            },
            "detailed_results": [
                {
                    "question": r.question,
                    "effectiveness_score": round(r.effectiveness_score, 3),
                    "faithfulness_score": round(r.faithfulness_score, 3),
                    "source_attribution_score": round(r.source_attribution_score, 3),
                    "overall_score": round(r.overall_score, 3)
                }
                for r in self.evaluation_results
            ],
            "recommendations": self._generate_recommendations(avg_effectiveness, avg_faithfulness, avg_source_attribution)
        }
        
        return report
    
    def _generate_recommendations(self, effectiveness: float, faithfulness: float, source_attribution: float) -> List[str]:
        recommendations = []
        
        if effectiveness < 0.7:
            recommendations.append("Improve answer relevance by fine-tuning the retrieval system or adjusting the prompt template")
        
        if faithfulness < 0.7:
            recommendations.append("Enhance faithfulness by improving the grounding of answers in source documents")
        
        if source_attribution < 0.7:
            recommendations.append("Improve source attribution by optimizing the retriever to find more relevant documents")
        
        if effectiveness >= 0.8 and faithfulness >= 0.8:
            recommendations.append("System is performing well. Consider expanding the knowledge base for broader coverage")
        
        if not recommendations:
            recommendations.append("Continue monitoring system performance and consider expanding evaluation criteria")
        
        return recommendations
    
    def save_results(self, filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        serializable_results = []
        for result in self.evaluation_results:
            serializable_results.append({
                "question": result.question,
                "answer": result.answer,
                "source_documents": result.source_documents,
                "effectiveness_score": result.effectiveness_score,
                "faithfulness_score": result.faithfulness_score,
                "source_attribution_score": result.source_attribution_score,
                "overall_score": result.overall_score,
                "timestamp": result.timestamp
            })
        
        data = {
            "evaluation_results": serializable_results,
            "report": self.generate_report(),
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Evaluation results saved to {filename}")
        return filename

def main():
    from rag_pipeline import NSWCrimeRAGPipeline
    
    pipeline = NSWCrimeRAGPipeline()
    documents_file = "data/rag_documents.json"
    
    if not os.path.exists("chroma_db"):
        print("Vector store not found. Initializing pipeline...")
        pipeline.initialize_pipeline(documents_file, force_recreate=True)
    else:
        pipeline.initialize_pipeline(documents_file, force_recreate=False)
    
    evaluator = RAGEvaluator(pipeline)
    results = evaluator.run_evaluation()
    
    report = evaluator.generate_report()
    
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    
    summary = report["evaluation_summary"]
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Average Effectiveness: {summary['average_effectiveness']}")
    print(f"Average Faithfulness: {summary['average_faithfulness']}")
    print(f"Average Source Attribution: {summary['average_source_attribution']}")
    print(f"Average Overall Score: {summary['average_overall_score']}")
    print(f"Unanswered Questions: {summary['unanswered_questions']} ({summary['unanswered_percentage']}%)")
    
    print("\nRecommendations:")
    for rec in report["recommendations"]:
        print(f"- {rec}")
    
    filename = evaluator.save_results()
    
    return report

if __name__ == "__main__":
    main()
