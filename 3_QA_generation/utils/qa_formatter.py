import random
from typing import List, Dict, Any, Union
import copy

class QAFormatter:
    """QA formatting utility class for handling question-answer pair formatting"""
    
    def __init__(self):
        self.choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    def format_single_question(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Formats a single question.
        
        Args:
            question: Original question dictionary.
            
        Returns:
            Formatted question dictionary.
        """
        formatted_question = copy.deepcopy(question)
        # if formatted_question['subcategory'] == 'breaking_state_comparison':
        #     import pdb;pdb.set_trace()
        
        # Add labels to choices
        if 'choices' in formatted_question:
            choices = formatted_question['choices']
            labeled_choices = []
            
            for i, choice in enumerate(choices):
                if i < len(self.choice_labels):
                    label = self.choice_labels[i]
                    # Add label as part of the choice string
                    labeled_choices.append(f"{label}. {choice}")
            
            formatted_question['choices'] = labeled_choices
            
            # Convert answer format
            correct_answer = formatted_question.get('correct_answer')
            if isinstance(correct_answer, int) and correct_answer < len(self.choice_labels):
                formatted_question['correct_answer'] = [self.choice_labels[correct_answer]]
            else:
                letter_answers = []
                for idx in correct_answer:
                    if isinstance(idx, int) and idx < len(self.choice_labels):
                        letter_answers.append(self.choice_labels[idx])
                # Sort answers
                letter_answers.sort()
                formatted_question['correct_answer'] = letter_answers

        # Set question type based on the number of correct answers
        if 'correct_answer' in formatted_question:
            if isinstance(formatted_question['correct_answer'], list):
                if len(formatted_question['correct_answer']) == 1:
                    formatted_question['question_type'] = 'single_choice'
                else:
                    formatted_question['question_type'] = 'multiple_choice'
            
        return formatted_question
    
    def balance_answer_distribution(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Balances the answer distribution to ensure that each option is chosen as the correct answer as evenly as possible.
        
        Args:
            questions: List of questions.
            
        Returns:
            List of questions with balanced answer distribution.
        """
        # Separate single-choice and multiple-choice questions
        single_choice_questions = [q for q in questions if q.get('question_type') == 'single_choice']
        other_questions = [q for q in questions if q.get('question_type') != 'single_choice']
        
        if not single_choice_questions:
            return questions
        
        # Count current answer distribution
        answer_counts = {}
        for question in single_choice_questions:
            choices = question.get('choices', [])
            if isinstance(choices, list) and len(choices) > 0:
                num_choices = len(choices)
                correct_answer = question.get('correct_answer')
                
                if num_choices not in answer_counts:
                    answer_counts[num_choices] = {i: 0 for i in range(num_choices)}
                
                if isinstance(correct_answer, int) and 0 <= correct_answer < num_choices:
                    answer_counts[num_choices][correct_answer] += 1
        
        # Reallocate answers to balance distribution
        balanced_questions = []
        
        for num_choices, counts in answer_counts.items():
            # Get all questions with this number of choices
            questions_with_n_choices = [
                q for q in single_choice_questions 
                if isinstance(q.get('choices'), list) and len(q.get('choices', [])) == num_choices
            ]
            
            if not questions_with_n_choices:
                continue
            
            # Calculate ideal distribution for each option
            total_questions = len(questions_with_n_choices)
            ideal_per_choice = total_questions // num_choices
            remainder = total_questions % num_choices
            
            # Create target distribution
            target_distribution = [ideal_per_choice] * num_choices
            for i in range(remainder):
                target_distribution[i] += 1
            
            # Reallocate answers
            shuffled_questions = questions_with_n_choices.copy()
            random.shuffle(shuffled_questions)
            
            question_idx = 0
            for choice_idx in range(num_choices):
                for _ in range(target_distribution[choice_idx]):
                    if question_idx < len(shuffled_questions):
                        # Rearrange choices so the correct answer is in the specified position
                        question = copy.deepcopy(shuffled_questions[question_idx])
                        question = self._rearrange_choices_for_answer(question, choice_idx)
                        balanced_questions.append(question)
                        question_idx += 1
        
        # Merge all questions
        return balanced_questions + other_questions
    
    def _rearrange_choices_for_answer(self, question: Dict[str, Any], target_answer_idx: int) -> Dict[str, Any]:
        """Rearranges choices so the correct answer is at the specified position.
        
        Args:
            question: Question dictionary.
            target_answer_idx: Target index for the correct answer.
            
        Returns:
            Question dictionary with rearranged choices.
        """
        choices = question.get('choices', [])
        current_answer_idx = question.get('correct_answer')
        
        if not isinstance(choices, list) or len(choices) <= target_answer_idx:
            return question
        
        if not isinstance(current_answer_idx, int) or current_answer_idx >= len(choices):
            return question
        
        # Create a new list of choices
        new_choices = choices.copy()
        
        # Move the correct answer to the target position
        correct_choice = choices[current_answer_idx]
        
        # Remove the correct answer from its current position
        new_choices.pop(current_answer_idx)
        
        # Insert the correct answer at the target position
        new_choices.insert(target_answer_idx, correct_choice)
        
        # Update the question
        question['choices'] = new_choices
        question['correct_answer'] = target_answer_idx
        
        return question
    
    def format_questions(self, questions: List[Dict[str, Any]], balance_distribution: bool = True) -> List[Dict[str, Any]]:
        """Formats a list of questions.
        
        Args:
            questions: List of questions.
            balance_distribution: Whether to balance the answer distribution.
            
        Returns:
            Formatted list of questions.
        """
        if not questions:
            return questions
            
        # Balance answer distribution
        if balance_distribution:
            questions = self.balance_answer_distribution(questions)
        
        # Format each question
        formatted_questions = []
        for question in questions:
            formatted_question = self.format_single_question(question)
            formatted_questions.append(formatted_question)
        
        return formatted_questions
    
    def format_qa_data(self, qa_data: Dict[str, Any], balance_distribution: bool = True) -> Dict[str, Any]:
        """Formats the entire QA data.
        
        Args:
            qa_data: Original QA data.
            balance_distribution: Whether to balance the answer distribution.
            
        Returns:
            Formatted QA data.
        """
        formatted_data = copy.deepcopy(qa_data)
        
        # Process single-state questions
        if 'single_state_questions' in formatted_data:
            for category, questions in formatted_data['single_state_questions'].items():
                if isinstance(questions, list):
                    # Balance answer distribution
                    if balance_distribution:
                        questions = self.balance_answer_distribution(questions)
                    
                    # Format each question
                    formatted_questions = []
                    for question in questions:
                        formatted_question = self.format_single_question(question)
                        formatted_questions.append(formatted_question)
                    
                    formatted_data['single_state_questions'][category] = formatted_questions
        
        # Process multi-state questions
        if 'multi_state_questions' in formatted_data:
            for category, questions in formatted_data['multi_state_questions'].items():
                if isinstance(questions, list):
                    # Balance answer distribution
                    if balance_distribution:
                        questions = self.balance_answer_distribution(questions)
                    
                    # Format each question
                    formatted_questions = []
                    for question in questions:
                        formatted_question = self.format_single_question(question)
                        formatted_questions.append(formatted_question)
                    
                    formatted_data['multi_state_questions'][category] = formatted_questions
        
        return formatted_data
    
    def get_answer_distribution_stats(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Gets answer distribution statistics.
        
        Args:
            questions: List of questions.
            
        Returns:
            Answer distribution statistics.
        """
        stats = {
            'single_choice': {},
            'multiple_choice': {},
            'total_questions': len(questions)
        }
        
        for question in questions:
            question_type = question.get('question_type')
            choices = question.get('choices', [])
            correct_answer = question.get('correct_answer')
            
            if question_type == 'single_choice' and isinstance(choices, list):
                num_choices = len(choices)
                if num_choices not in stats['single_choice']:
                    stats['single_choice'][num_choices] = {i: 0 for i in range(num_choices)}
                
                if isinstance(correct_answer, int) and 0 <= correct_answer < num_choices:
                    stats['single_choice'][num_choices][correct_answer] += 1
            
            elif question_type == 'multiple_choice':
                if isinstance(correct_answer, list):
                    answer_count = len(correct_answer)
                    if answer_count not in stats['multiple_choice']:
                        stats['multiple_choice'][answer_count] = 0
                    stats['multiple_choice'][answer_count] += 1
        
        return stats