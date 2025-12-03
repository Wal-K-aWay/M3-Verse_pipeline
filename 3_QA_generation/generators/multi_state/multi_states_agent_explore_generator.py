import os
import json
import random
import inspect
from typing import Dict, List, Any
from ..base_generator import BaseGenerator

class MultiStateAgentExploreGenerator(BaseGenerator):    
    def __init__(self, scene_path: str):
        super().__init__(scene_path)
        template_path = os.path.join(os.path.dirname(__file__), '../../templates/multi_state/agent_explore_template.json')
        with open(template_path, 'r', encoding='utf-8') as f:
            self.templates = json.load(f)['agent_explore']
        
    def generate_questions(self) -> List[Dict[str, Any]]:
        questions = []
        
        states = self._get_ordered_states()
        
        for i in range(len(states) - 1):
            
            state1 = states[i]
            state2 = states[i + 1]

            questions.extend(self._generate_room_visit_time_questions(state1, state2))
            questions.extend(self._generate_room_visit_count_questions(state1, state2))
            questions.extend(self._generate_room_visit_change_questions(state1, state2))
            questions.extend(self._generate_nth_mth_room_comparison_questions(state1, state2))
        
        return questions


    def _generate_longest_room_visit_time_questions(self, state1: str, state2: str, available_rooms: Dict[str, Any], sorted_room_counts: List[tuple], all_observed_room_names: List[str], room_stay_frames: Dict[str, Dict[str, List]], trajectory1: List[Dict[str, Any]], trajectory2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        longest_stay_question_template = self.templates['longest_stay_room']['question_template']
        questions = []

        len_video1 = len(trajectory1)
        len_video2 = len(trajectory2)

        most_observed_id = sorted_room_counts[-1][0]
        most_observed_name = available_rooms[most_observed_id]
        random.shuffle(all_observed_room_names)
        choices = [most_observed_name]
        num_choices = random.randint(4, 6)
        for obj in all_observed_room_names:
            if obj not in choices and len(choices) < num_choices:
                choices.append(obj)
        random.shuffle(choices)
        correct_choice = choices.index(most_observed_name)
        
        related_frames = {state1: set(), state2: set()}
        answer_source = []
        answer_source.append({'type': 'text', 'content': f'The {self.get_video_num(state1)} video has {len_video1} frames and the {self.get_video_num(state2)} video has {len_video2} frames.'})
        for choice in choices:
            room_id = None
            for rid, rname in available_rooms.items():
                if rname == choice:
                    room_id = rid
                    break
            
            if room_id and room_id in room_stay_frames:
                room_frames = room_stay_frames[room_id]
                answer_source.append({'type': 'text', 'content': f'Agent visited {choice} in the following frames:'})
                
                if state1 in room_frames and room_frames[state1]:
                    frames_state1 = room_frames[state1]
                    related_frames[state1].update(frames_state1)
                    clips = self.find_continous_clips(frames_state1)
                    answer_source.append({'type': 'text', 'content': f'In the {self.get_video_num(state1)} video:'})
                    for i, clip in enumerate(clips):
                        start_frame = clip[0]
                        end_frame = clip[-1]
                        if i == len(clip) - 1:
                            answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                        else:
                            answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
                
                if state2 in room_frames and room_frames[state2]:
                    frames_state2 = room_frames[state2]
                    related_frames[state2].update(frames_state2)
                    clips = self.find_continous_clips(frames_state2)
                    answer_source.append({'type': 'text', 'content': f'In the {self.get_video_num(state2)} video:'})
                    for clip in clips:
                        start_frame = clip[0]
                        end_frame = clip[-1]
                        if i == len(clip) - 1:
                            answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                        else:
                            answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
        related_frames[state1] = sorted(list(related_frames[state1]))
        related_frames[state2] = sorted(list(related_frames[state2]))

        question_text = longest_stay_question_template.format(state1=state1, state2=state2)
        questions.append({
            'question': question_text,
            'choices': choices,
            'correct_answer': correct_choice,
            'question_type': self.templates['longest_stay_room']['question_type'],
            'category': 'multi_states_agent_explore',
            'subcategory': 'longest_stay_room',
            'capabilities': self.templates['longest_stay_room']['capabilities'],
            'answer_source': answer_source,
            'related_frames': related_frames,
            'hallucination': False
        })

        return questions

    def _generate_shortest_room_visit_time_questions(self, state1: str, state2: str, available_rooms: Dict[str, Any], sorted_room_counts: List[tuple], all_observed_room_names: List[str], room_stay_frames: Dict[str, Dict[str, List]], trajectory1: List[Dict[str, Any]], trajectory2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        shortest_stay_question_template = self.templates['shortest_stay_room']['question_template']
        questions = []

        len_video1 = len(trajectory1)
        len_video2 = len(trajectory2)
        
        least_observed_id = sorted_room_counts[0][0]
        least_observed_name = available_rooms[least_observed_id]
        random.shuffle(all_observed_room_names)
        choices = [least_observed_name]
        num_choices = random.randint(4, 6)
        for obj in all_observed_room_names:
            if obj not in choices and len(choices) < num_choices:
                choices.append(obj)
        random.shuffle(choices)
        correct_choice = choices.index(least_observed_name)
        
        related_frames = {state1: set(), state2: set()}
        answer_source = []
        answer_source.append({'type': 'text', 'content': f'The {self.get_video_num(state1)} video has {len_video1} frames and the {self.get_video_num(state2)} video has {len_video2} frames.'})
        for choice in choices:
            room_id = None
            for rid, rname in available_rooms.items():
                if rname == choice:
                    room_id = rid
                    break
            
            if room_id and room_id in room_stay_frames:
                room_frames = room_stay_frames[room_id]
                answer_source.append({'type': 'text', 'content': f'Agent visited {choice} in the following frames:'})
                
                if state1 in room_frames and room_frames[state1]:
                    frames_state1 = room_frames[state1]
                    related_frames[state1].update(frames_state1)
                    clips = self.find_continous_clips(frames_state1)
                    answer_source.append({'type': 'text', 'content': f'In the {self.get_video_num(state1)} video:'})
                    for i, clip in enumerate(clips):
                        start_frame = clip[0]
                        end_frame = clip[-1]
                        if i == len(clip) - 1:
                            answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                        else:
                            answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
                
                if state2 in room_frames and room_frames[state2]:
                    frames_state2 = room_frames[state2]
                    related_frames[state2].update(frames_state2)
                    clips = self.find_continous_clips(frames_state2)
                    answer_source.append({'type': 'text', 'content': f'In the {self.get_video_num(state2)} video:'})
                    for clip in clips:
                        start_frame = clip[0]
                        end_frame = clip[-1]
                        if i == len(clip) - 1:
                            answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                        else:
                            answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
        related_frames[state1] = sorted(list(related_frames[state1]))
        related_frames[state2] = sorted(list(related_frames[state2]))

        question_text = shortest_stay_question_template.format(state1=state1, state2=state2)
        questions.append({
            'question': question_text,
            'choices': choices,
            'correct_answer': correct_choice,
            'question_type': self.templates['shortest_stay_room']['question_type'],
            'category': 'multi_states_agent_explore',
            'subcategory': 'shortest_stay_room',
            'capabilities': self.templates['shortest_stay_room']['capabilities'],
            'answer_source': answer_source,
            'related_frames': related_frames,
            'hallucination': False
        })
        return questions

    def _generate_cross_state_longest_room_comparison_questions(self, state1: str, state2: str, combination_list: List[tuple], trajectory1: List[Dict[str, Any]], trajectory2: List[Dict[str, Any]]):
        question_text = self.templates['cross_state_longest_room_comparison']['question_template']
        questions = []

        len_video1 = len(trajectory1)
        len_video2 = len(trajectory2)

        used_combinations = set()
        while True:
            available_candidates = [combo for combo in combination_list if combo[0] not in used_combinations]
            
            if len(available_candidates) < 4:
                break
            
            num_choices = min(random.randint(4, 6), len(available_candidates))
            selected_candidates = random.sample(available_candidates, num_choices)
            selected_candidates.sort(key=lambda x: len(x[1]))
            longest_combo = selected_candidates[-1]
            max_duration = len(longest_combo[1])
            longest_combos = [combo for combo in selected_candidates if len(combo[1]) == max_duration]
            choices = [combo[0] for combo in selected_candidates]
            random.shuffle(choices)
            correct_answer = [choices.index(combo[0]) for combo in longest_combos]
            
            related_frames = {state1: set(), state2: set()}
            answer_source = []
            answer_source.append({'type': 'text', 'content': f'The {self.get_video_num(state1)} video has {len_video1} frames and the {self.get_video_num(state2)} video has {len_video2} frames.'})
            for combo in selected_candidates:
                choice_text, choice_frames, room_name, choice_state = combo
                
                related_frames[choice_state].update(choice_frames)
                answer_source.append({'type': 'text', 'content': f'Agent visited {room_name} in the following frames:'})
                clips = self.find_continous_clips(choice_frames)
                answer_source.append({'type': 'text', 'content': f'In the {self.get_video_num(choice_state)} video:'})
                for i, clip in enumerate(clips):
                    start_frame = clip[0]
                    end_frame = clip[-1]
                    if i == len(clips) - 1:
                        answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                    else:
                        answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
            related_frames[state1] = sorted(list(related_frames[state1]))
            related_frames[state2] = sorted(list(related_frames[state2]))

            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['cross_state_longest_room_comparison']['question_type'],
                'category': 'multi_states_agent_explore',
                'subcategory': 'cross_state_longest_room_comparison',
                'capabilities': self.templates['cross_state_longest_room_comparison']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })
            
            for combo in selected_candidates:
                used_combinations.add(combo[0])

        return questions
    
    def _generate_cross_state_shortest_room_comparison_questions(self, state1: str, state2: str, combination_list: List[tuple], trajectory1: List[Dict[str, Any]], trajectory2: List[Dict[str, Any]]):
        question_text = self.templates['cross_state_shortest_room_comparison']['question_template']
        questions = []

        len_video1 = len(trajectory1)
        len_video2 = len(trajectory2)

        used_combinations = set()
        while True:
            available_candidates = [combo for combo in combination_list if combo[0] not in used_combinations]
            
            if len(available_candidates) < 4:
                break
            
            num_choices = min(random.randint(4, 6), len(available_candidates))
            selected_candidates = random.sample(available_candidates, num_choices)
            selected_candidates.sort(key=lambda x: len(x[1]))
            shortest_combo = selected_candidates[0] 
            min_duration = len(shortest_combo[1])
            shortest_combos = [combo for combo in selected_candidates if len(combo[1]) == min_duration]
            choices = [combo[0] for combo in selected_candidates]
            random.shuffle(choices)
            correct_answer = [choices.index(combo[0]) for combo in shortest_combos]
            
            related_frames = {state1: set(), state2: set()}
            answer_source = []
            answer_source.append({'type': 'text', 'content': f'The {self.get_video_num(state1)} video has {len_video1} frames and the {self.get_video_num(state2)} video has {len_video2} frames.'})
            for combo in selected_candidates:
                choice_text, choice_frames, room_name, choice_state = combo
                
                related_frames[choice_state].update(choice_frames)
                answer_source.append({'type': 'text', 'content': f'Agent visited {room_name} in the following frames:'})
                clips = self.find_continous_clips(choice_frames)
                answer_source.append({'type': 'text', 'content': f'In the {self.get_video_num(choice_state)} video:'})
                for i, clip in enumerate(clips):
                    start_frame = clip[0]
                    end_frame = clip[-1]
                    if i == len(clips) - 1:
                        answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                    else:
                        answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
            related_frames[state1] = sorted(list(related_frames[state1]))
            related_frames[state2] = sorted(list(related_frames[state2]))

            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['cross_state_shortest_room_comparison']['question_type'],
                'category': 'multi_states_agent_explore',
                'subcategory': 'cross_state_shortest_room_comparison',
                'capabilities': self.templates['cross_state_shortest_room_comparison']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })
            
            for combo in selected_candidates:
                used_combinations.add(combo[0])

        return questions

    def _generate_room_visit_time_questions(self, state1: str, state2: str) -> List[Dict[str, Any]]:
        available_keys = ['type', 'shape', 'unique_object']

        questions = []

        rooms = self.scene_data['room_static']['room_static_details']
        available_rooms = {}
        for room in rooms:
            room_id = room['room_name']
            room_name = None
            if room_id in self.room_names[state1]:
                room_names = self.room_names[state1][room_id]
                room_name, room_key = self._select_name(room_names, available_keys, )
            if not room_name:
                continue
            available_rooms[room['room_name']] = room_name
        
        if len(available_rooms) < 2:
            return questions

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']

        room_stay_frames = {}
        
        room_stay_frames_state1 = {}
        for i, step in enumerate(trajectory1):
            current_room_name = step['room_name']
            if current_room_name in available_rooms:
                if current_room_name not in room_stay_frames_state1:
                    room_stay_frames_state1[current_room_name] = []
                room_stay_frames_state1[current_room_name].append(i)
                
                if current_room_name not in room_stay_frames:
                    room_stay_frames[current_room_name] = {}
                if state1 not in room_stay_frames[current_room_name]:
                    room_stay_frames[current_room_name][state1] = []
                
                room_stay_frames[current_room_name][state1].append(i)

        room_stay_frames_state2 = {}
        for i, step in enumerate(trajectory2):
            current_room_name = step['room_name']
            if current_room_name in available_rooms:
                if current_room_name not in room_stay_frames_state2:
                    room_stay_frames_state2[current_room_name] = []
                room_stay_frames_state2[current_room_name].append(i)
                
                if current_room_name not in room_stay_frames:
                    room_stay_frames[current_room_name] = {}
                if state2 not in room_stay_frames[current_room_name]:
                    room_stay_frames[current_room_name][state2] = []
                room_stay_frames[current_room_name][state2].append(i)
        
        if len(room_stay_frames) < 2:
            return questions
            
        room_total_frames = {}
        for room_id, states_data in room_stay_frames.items():
            total_frames = 0
            for state_frames in states_data.values():
                total_frames += len(state_frames)
            room_total_frames[room_id] = total_frames
        
        sorted_room_counts = sorted(room_total_frames.items(), key=lambda item: item[1])

        all_observed_room_names = [available_rooms[obj_id] for obj_id in room_stay_frames.keys() if obj_id in available_rooms]

        questions.extend(self._generate_longest_room_visit_time_questions(state1, 
                                                                          state2, 
                                                                          available_rooms, 
                                                                          sorted_room_counts, 
                                                                          all_observed_room_names, 
                                                                          room_stay_frames,
                                                                          trajectory1, 
                                                                          trajectory2))

        questions.extend(self._generate_shortest_room_visit_time_questions(state1, 
                                                                          state2, 
                                                                          available_rooms, 
                                                                          sorted_room_counts, 
                                                                          all_observed_room_names, 
                                                                          room_stay_frames,
                                                                          trajectory1, 
                                                                          trajectory2))

        if len(room_stay_frames_state1) > 0 and len(room_stay_frames_state2) > 0:
            state_room_combinations = {}
            
            for room_id, frames in room_stay_frames_state1.items():
                if room_id in available_rooms:
                    room_name = available_rooms[room_id]
                    key = (room_name, state1)
                    state_room_combinations[key] = frames
            
            for room_id, frames in room_stay_frames_state2.items():
                if room_id in available_rooms:
                    room_name = available_rooms[room_id]
                    key = (room_name, state2)
                    state_room_combinations[key] = frames

            combination_list = [(f"{room} at {self.get_video_num(state)} state", frames, room, state) for (room, state), frames in state_room_combinations.items()]
            
            if len(combination_list) >= 4:
                combination_list.sort(key=lambda x: len(x[1]))
                
                questions.extend(self._generate_cross_state_longest_room_comparison_questions(state1,
                                                                                              state2,
                                                                                             combination_list,
                                                                                             trajectory1,
                                                                                             trajectory2))


                questions.extend(self._generate_cross_state_shortest_room_comparison_questions(state1, 
                                                                                               state2, 
                                                                                               combination_list, 
                                                                                               trajectory1, 
                                                                                               trajectory2))
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions


    def _generate_room_visit_count_questions(self, state1: str, state2: str) -> List[Dict[str, Any]]:
        available_keys = ['type', 'biggest_room', 'smallest_room', 'biggest_type', 'smallest_type', 'shape']
        question_template = self.templates['total_room_visits']['question_template']
        questions = []
        
        # Get available rooms
        rooms = self.scene_data['room_static']['room_static_details']
        available_rooms = {}
        for room in rooms:
            room_id = room['room_name']
            room_name = None
            if room_id in self.room_names[state1]:
                room_names = self.room_names[state1][room_id]
                room_name, room_key = self._select_name(room_names, available_keys)
            if not room_name:
                continue
            available_rooms[room['room_name']] = room_name
        
        if len(available_rooms) < 1:
            return questions
        
        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        len_video1= len(trajectory1)
        len_video2= len(trajectory2)
        
        # Track room visits and frames for each state
        room_visit_data = {}
        
        # Track visits in state1
        current_room_id = None
        visit_start_frame = None
        for i, step in enumerate(trajectory1):
            room_id = step['room_name']
            if room_id in available_rooms:
                if current_room_id != room_id:
                    # New visit detected
                    if room_id not in room_visit_data:
                        room_visit_data[room_id] = {state1: []}
                    elif state1 not in room_visit_data[room_id]:
                        room_visit_data[room_id][state1] = []
                    if visit_start_frame is not None and current_room_id is not None:
                        room_visit_data[current_room_id][state1].append((visit_start_frame, i))
                    current_room_id = room_id
                    visit_start_frame = i
        if visit_start_frame is not None and current_room_id is not None:
            room_visit_data[current_room_id][state1].append((visit_start_frame, len(trajectory1)))
        
        # Track visits in state2
        current_room_id = None
        visit_start_frame = None
        for i, step in enumerate(trajectory2):
            room_id = step['room_name']
            if room_id in available_rooms:
                if current_room_id != room_id:
                    # New visit detected
                    if room_id not in room_visit_data:
                        room_visit_data[room_id] = {state2: []}
                    elif state2 not in room_visit_data[room_id]:
                        room_visit_data[room_id][state2] = []
                    if visit_start_frame is not None and current_room_id is not None:
                        room_visit_data[current_room_id][state2].append((visit_start_frame, i))

                    current_room_id = room_id
                    visit_start_frame = i
        if visit_start_frame is not None and current_room_id is not None:
            room_visit_data[current_room_id][state2].append((visit_start_frame, len(trajectory2)))
        
        # Generate questions for each room
        for room_id, visits_data in room_visit_data.items():
            room_name = available_rooms[room_id]
            state1_visits = visits_data.get(state1, [])
            state2_visits = visits_data.get(state2, [])
            total_visits = len(state1_visits) + len(state2_visits)
            
            # Generate choices around the correct answer
            num_choices = random.randint(4, 6)
            choices = set([total_visits])  # Include correct answer
            
            # Generate other plausible visit counts using a normal distribution
            mean = total_visits
            std_dev = max(1, total_visits // 2)  # Standard deviation based on total_visits
            
            attempts = 0
            max_attempts = 50
            while len(choices) < num_choices and attempts < max_attempts:
                # Generate a candidate using normal distribution
                candidate = int(random.normalvariate(mean, std_dev))
                candidate = max(1, candidate)  # Ensure candidate is at least 1
                choices.add(candidate)
                attempts += 1
            
            # If we still don't have enough choices, fill with sequential numbers
            if len(choices) < num_choices:
                for i in range(1, 20):
                    if len(choices) >= num_choices:
                        break
                    choices.add(i)
            
            choices = list(choices)
            original_correct_answer = total_visits
            correct_choice_index = choices.index(original_correct_answer)

            if random.random() < 1/3:
                is_hallucination = True
                correct_answer_value = self.no_valid_option
                choices.pop(correct_choice_index)
                choices.append(correct_answer_value)
            else:
                is_hallucination = False
                correct_answer_value = original_correct_answer
                num_current_choices = len(choices)
                all_indices = list(range(num_current_choices))
                candidate_remove_ids = [idx for idx in all_indices if idx != correct_choice_index]
                if candidate_remove_ids:
                    remove_id = random.choice(candidate_remove_ids)
                    choices.pop(remove_id)
                choices.append(self.no_valid_option)

            random.shuffle(choices)
            correct_choice = choices.index(correct_answer_value)
            
            # Generate related_frames and answer_source
            related_frames = {state1: set(), state2: set()}
            answer_source = []
            answer_source.append({'type': 'text', 'content': f'The {self.get_video_num(state1)} video has {len_video1} frames and the {self.get_video_num(state2)} video has {len_video2} frames.'})
            
            if state1_visits:
                answer_source.append({'type': 'text', 'content': f'In the {self.get_video_num(state1)} video, agent visited {room_name} {len(state1_visits)} times:'})
                for i, (start_frame, end_frame) in enumerate(state1_visits):
                    for frame in range(start_frame, end_frame + 1):
                        related_frames[state1].add(frame)
                    if i == len(state1_visits) - 1:
                        answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                    else:
                        answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
            else:
                answer_source.append({'type': 'text', 'content': f'In the {self.get_video_num(state1)} video, agent did not visit {room_name}.'})
            if state2_visits:
                answer_source.append({'type': 'text', 'content': f'In the {self.get_video_num(state2)} video, agent visited {room_name} {len(state2_visits)} times:'})
                for i, (start_frame, end_frame) in enumerate(state2_visits):
                    for frame in range(start_frame, end_frame + 1):
                        related_frames[state2].add(frame)
                    if i == len(state2_visits) - 1:
                        answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                    else:
                        answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
            else:
                answer_source.append({'type': 'text', 'content': f'In the {self.get_video_num(state2)} video, agent did not visit {room_name}.'})
            answer_source.append({'type': 'text', 'content': f'Total visits to {room_name} {total_visits} times'})
            related_frames[state1] = sorted(list(related_frames[state1]))
            related_frames[state2] = sorted(list(related_frames[state2]))

            question_text = question_template.format(room=room_name, state1=state1, state2=state2)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_choice,
                'question_type': self.templates['total_room_visits']['question_type'],
                'category': 'multi_states_agent_explore',
                'subcategory': 'total_room_visits',
                'capabilities': self.templates['total_room_visits']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    

    def _generate_only_state1_visible_questions(self, state1: str, state2: str, union_rooms: List[str], inter_rooms: List[str], only_state1_rooms: List[str], only_state2_rooms: List[str], rooms_visited_state1: Dict[str, tuple[str, List]], rooms_visited_state2: Dict[str, tuple[str, List]]) -> List[Dict[str, Any]]:
        question_template = self.templates['rooms_visited_only_state1']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        len_video1 = len(trajectory1)
        len_video2 = len(trajectory2)

        used_rooms = set()
        all_both_visited_count = 0
        other_answer_count = 0
        
        while True:
            available_candidates = [room for room in union_rooms if room not in used_rooms]
            
            if len(available_candidates) < 3:
                break

            related_frames = {state1: set(), state2: set()}
            answer_source = [{'type': 'text', 'content': f'The {self.get_video_num(state1)} video has {len_video1} frames and the {self.get_video_num(state2)} video has {len_video2} frames.'}]

            num_choices = min(random.randint(3, 5), len(available_candidates))
            choice_ids = random.sample(available_candidates, num_choices)
            choices = []
            correct_choices = []
            for id in choice_ids:
                if id in only_state1_rooms:
                    room_name = rooms_visited_state1[id][0]
                    choices.append(room_name)
                    correct_choices.append(room_name)
                    related_frames[state1].update(rooms_visited_state1[id][1])
                    # answer_source.append({'type': 'text', 'content': f'Agent visited {room_name} in the following clips in the {self.get_video_num(state1)} video: '})
                    # clips = self.find_continous_clips(rooms_visited_state1[id][1])
                    # for i, clip in enumerate(clips):
                    #     start_frame = clip[0]
                    #     end_frame = clip[-1]
                    #     if i == len(clips) - 1:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                    #     else:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
                elif id in inter_rooms:
                    room_name = rooms_visited_state1[id][0]
                    choices.append(room_name)
                    related_frames[state1].update(rooms_visited_state1[id][1])
                    related_frames[state2].update(rooms_visited_state2[id][1])
                    # answer_source.append({'type': 'text', 'content': f'Agent visited {room_name} in the following clips: '})
                    # clips = self.find_continous_clips(rooms_visited_state1[id][1])
                    # answer_source.append({'type': 'text', 'content': f'In the {self.get_video_num(state1)} video: '})
                    # for i, clip in enumerate(clips):
                    #     start_frame = clip[0]
                    #     end_frame = clip[-1]
                    #     if i == len(clips) - 1:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                    #     else:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
                    # answer_source.append({'type': 'text', 'content': f'In the {self.get_video_num(state2)} video: '})
                    # clips = self.find_continous_clips(rooms_visited_state2[id][1])
                    # for i, clip in enumerate(clips):
                    #     start_frame = clip[0]
                    #     end_frame = clip[-1]
                    #     if i == len(clips) - 1:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                    #     else:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
                else:
                    room_name = rooms_visited_state2[id][0]
                    choices.append(room_name)
                    related_frames[state2].update(rooms_visited_state2[id][1])
                    # answer_source.append({'type': 'text', 'content': f'Agent visited {room_name} in the following clips in the {self.get_video_num(state2)} video: '})
                    # clips = self.find_continous_clips(rooms_visited_state2[id][1])
                    # for i, clip in enumerate(clips):
                    #     start_frame = clip[0]
                    #     end_frame = clip[-1]
                    #     if i == len(clips) - 1:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                    #     else:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
            choices.append(self.no_valid_option)
            random.shuffle(choices)
            
            if correct_choices:
                correct_indices = [choices.index(name) for name in correct_choices if name in choices]
                is_hallucination = False
            else:
                correct_indices = [choices.index(self.no_valid_option)]
                is_hallucination = True

            should_add_question = True
            if is_hallucination:
                if all_both_visited_count >= other_answer_count // 2:
                    should_add_question = False
                else:
                    all_both_visited_count += 1
            else:
                other_answer_count += 1

            if should_add_question:
                related_frames[state1] = sorted(list(related_frames[state1]))
                related_frames[state2] = sorted(list(related_frames[state2]))

                question_text = question_template.format(state1=state1, state2=state2)
                questions.append({
                    'question': question_text,
                    'choices': choices,
                    'correct_answer': correct_indices,
                    'question_type': self.templates['rooms_visited_only_state1']['question_type'],
                    'category': 'multi_states_agent_explore',
                    'subcategory': 'rooms_visited_only_state1',
                    'capabilities': self.templates['rooms_visited_only_state1']['capabilities'],
                    'answer_source': answer_source,
                    'related_frames': related_frames,
                    'hallucination': is_hallucination
                })

            for room in choice_ids:
                used_rooms.add(room)
        
        return questions

    def _generate_only_state2_visible_questions(self, state1: str, state2: str, union_rooms: List[str], inter_rooms: List[str], only_state1_rooms: List[str], only_state2_rooms: List[str], rooms_visited_state1: Dict[str, tuple[str, List]], rooms_visited_state2: Dict[str, tuple[str, List]]) -> List[Dict[str, Any]]:
        question_template = self.templates['rooms_visited_only_state2']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        len_video1 = len(trajectory1)
        len_video2 = len(trajectory2)

        used_rooms = set()
        all_both_visited_count = 0
        other_answer_count = 0
        
        while True:
            available_candidates = [room for room in union_rooms if room not in used_rooms]
            
            if len(available_candidates) < 3:
                break

            related_frames = {state1: set(), state2: set()}
            answer_source = [{'type': 'text', 'content': f'The {self.get_video_num(state1)} video has {len_video1} frames and the {self.get_video_num(state2)} video has {len_video2} frames.'}]

            num_choices = min(random.randint(3, 5), len(available_candidates))
            choice_ids = random.sample(available_candidates, num_choices)
            choices = []
            correct_choices = []
            for id in choice_ids:
                if id in only_state2_rooms:
                    room_name = rooms_visited_state2[id][0]
                    choices.append(room_name)
                    correct_choices.append(room_name)
                    related_frames[state2].update(rooms_visited_state2[id][1])
                    # answer_source.append({'type': 'text', 'content': f'Agent visited {room_name} in the following clips in the {self.get_video_num(state2)} video: '})
                    # clips = self.find_continous_clips(rooms_visited_state2[id][1])
                    # for i, clip in enumerate(clips):
                    #     start_frame = clip[0]
                    #     end_frame = clip[-1]
                    #     if i == len(clips) - 1:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                    #     else:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
                elif id in inter_rooms:
                    room_name = rooms_visited_state2[id][0]
                    choices.append(room_name)
                    related_frames[state1].update(rooms_visited_state1[id][1])
                    related_frames[state2].update(rooms_visited_state2[id][1])
                    # answer_source.append({'type': 'text', 'content': f'Agent visited {room_name} in the following clips: '})
                    # clips = self.find_continous_clips(rooms_visited_state2[id][1])
                    # answer_source.append({'type': 'text', 'content': f'In the {self.get_video_num(state2)} video: '})
                    # for i, clip in enumerate(clips):
                    #     start_frame = clip[0]
                    #     end_frame = clip[-1]
                    #     if i == len(clips) - 1:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                    #     else:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
                    # answer_source.append({'type': 'text', 'content': f'In the {self.get_video_num(state1)} video: '})
                    # clips = self.find_continous_clips(rooms_visited_state1[id][1])
                    # for i, clip in enumerate(clips):
                    #     start_frame = clip[0]
                    #     end_frame = clip[-1]
                    #     if i == len(clips) - 1:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                    #     else:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
                else:
                    room_name = rooms_visited_state1[id][0]
                    choices.append(room_name)
                    related_frames[state1].update(rooms_visited_state1[id][1])
                    # answer_source.append({'type': 'text', 'content': f'Agent visited {room_name} in the following clips in the {self.get_video_num(state1)} video: '})
                    # clips = self.find_continous_clips(rooms_visited_state1[id][1])
                    # for i, clip in enumerate(clips):
                    #     start_frame = clip[0]
                    #     end_frame = clip[-1]
                    #     if i == len(clips) - 1:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                    #     else:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
            choices.append(self.no_valid_option)
            random.shuffle(choices)
            if correct_choices:
                correct_indices = [choices.index(name) for name in correct_choices if name in choices]
                is_hallucination = False
            else:
                correct_indices = [choices.index(self.no_valid_option)]
                is_hallucination = True

            should_add_question = True
            if is_hallucination:
                if all_both_visited_count >= other_answer_count // 2:
                    should_add_question = False
                else:
                    all_both_visited_count += 1
            else:
                other_answer_count += 1

            if should_add_question:
                related_frames[state1] = sorted(list(related_frames[state1]))
                related_frames[state2] = sorted(list(related_frames[state2]))

                question_text = question_template.format(state1=state1, state2=state2)
                questions.append({
                    'question': question_text,
                    'choices': choices,
                    'correct_answer': correct_indices,
                    'question_type': self.templates['rooms_visited_only_state2']['question_type'],
                    'category': 'multi_states_agent_explore',
                    'subcategory': 'rooms_visited_only_state2',
                    'capabilities': self.templates['rooms_visited_only_state2']['capabilities'],
                    'answer_source': answer_source,
                    'related_frames': related_frames,
                    'hallucination': is_hallucination
                })

            for room in choice_ids:
                used_rooms.add(room)
        
        return questions

    def _generate_rooms_visited_both_states_questions(self, state1: str, state2: str, union_rooms: List[str], inter_rooms: List[str], only_state1_rooms: List[str], only_state2_rooms: List[str], rooms_visited_state1: Dict[str, tuple[str, List]], rooms_visited_state2: Dict[str, tuple[str, List]]) -> List[Dict[str, Any]]:
        question_template = self.templates['rooms_visited_both_states']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        len_video1 = len(trajectory1)
        len_video2 = len(trajectory2)

        used_rooms = set()
        hallucination_count = 0
        not_hallucination_count = 0
        
        while True:
            available_candidates = [room for room in union_rooms if room not in used_rooms]
            
            if len(available_candidates) < 3:
                break

            related_frames = {state1: set(), state2: set()}
            answer_source = [{'type': 'text', 'content': f'The {self.get_video_num(state1)} video has {len_video1} frames and the {self.get_video_num(state2)} video has {len_video2} frames.'}]

            num_choices = min(random.randint(3, 5), len(available_candidates))
            choice_ids = random.sample(available_candidates, num_choices)
            choices = []
            correct_choices = []
            for id in choice_ids:
                if id in inter_rooms:
                    # 房间在两个状态都被访问
                    room_name = rooms_visited_state1[id][0]
                    choices.append(room_name)
                    correct_choices.append(room_name)
                    related_frames[state1].update(rooms_visited_state1[id][1])
                    related_frames[state2].update(rooms_visited_state2[id][1])
                    # answer_source.append({'type': 'text', 'content': f'Agent visited {room_name} in the following clips: '})
                    # clips = self.find_continous_clips(rooms_visited_state1[id][1])
                    # answer_source.append({'type': 'text', 'content': f'In the {self.get_video_num(state1)} video: '})
                    # for i, clip in enumerate(clips):
                    #     start_frame = clip[0]
                    #     end_frame = clip[-1]
                    #     if i == len(clips) - 1:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                    #     else:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
                    # answer_source.append({'type': 'text', 'content': f'In the {self.get_video_num(state2)} video: '})
                    # clips = self.find_continous_clips(rooms_visited_state2[id][1])
                    # for i, clip in enumerate(clips):
                    #     start_frame = clip[0]
                    #     end_frame = clip[-1]
                    #     if i == len(clips) - 1:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                    #     else:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
                elif id in only_state1_rooms:
                    # 房间仅在state1被访问
                    room_name = rooms_visited_state1[id][0]
                    choices.append(room_name)
                    related_frames[state1].update(rooms_visited_state1[id][1])
                    # answer_source.append({'type': 'text', 'content': f'Agent visited {room_name} in the following clips in the {self.get_video_num(state1)} video: '})
                    # clips = self.find_continous_clips(rooms_visited_state1[id][1])
                    # for i, clip in enumerate(clips):
                    #     start_frame = clip[0]
                    #     end_frame = clip[-1]
                    #     if i == len(clips) - 1:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                    #     else:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
                else:
                    # 房间仅在state2被访问
                    room_name = rooms_visited_state2[id][0]
                    choices.append(room_name)
                    related_frames[state2].update(rooms_visited_state2[id][1])
                    # answer_source.append({'type': 'text', 'content': f'Agent visited {room_name} in the following clips in the {self.get_video_num(state2)} video: '})
                    # clips = self.find_continous_clips(rooms_visited_state2[id][1])
                    # for i, clip in enumerate(clips):
                    #     start_frame = clip[0]
                    #     end_frame = clip[-1]
                    #     if i == len(clips) - 1:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}.'})
                    #     else:
                    #         answer_source.append({'type': 'text', 'content': f'frame {start_frame} to frame {end_frame}, '})
            
            choices.append(self.no_valid_option)
            random.shuffle(choices)
            
            if correct_choices:
                correct_indices = [choices.index(name) for name in correct_choices if name in choices]
                is_hallucination = False
            else:
                correct_indices = [choices.index(self.no_valid_option)]
                is_hallucination = True

            should_add_question = True
            if is_hallucination:
                if not_hallucination_count >= hallucination_count // 2:
                    should_add_question = False
                else:
                    hallucination_count += 1
            else:
                not_hallucination_count += 1

            if should_add_question:
                related_frames[state1] = sorted(list(related_frames[state1]))
                related_frames[state2] = sorted(list(related_frames[state2]))

                question_text = question_template.format(state1=state1, state2=state2)
                questions.append({
                    'question': question_text,
                    'choices': choices,
                    'correct_answer': correct_indices,
                    'question_type': self.templates['rooms_visited_both_states']['question_type'],
                    'category': 'multi_states_agent_explore',
                    'subcategory': 'rooms_visited_both_states',
                    'capabilities': self.templates['rooms_visited_both_states']['capabilities'],
                    'answer_source': answer_source,
                    'related_frames': related_frames,
                    'hallucination': is_hallucination
                })

            for room in choice_ids:
                used_rooms.add(room)
        
        return questions

    def _generate_room_visit_change_questions(self, state1: str, state2: str) -> List[Dict[str, Any]]:
        available_keys = ['type', 'biggest_room', 'smallest_room', 'biggest_type', 'smallest_type', 'shape']

        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        
        rooms_visited_state1 = {}
        for i, step in enumerate(trajectory1):
            current_room_name = step['room_name']
            if current_room_name in self.room_names[state1]:
                room_names_dict = self.room_names[state1][current_room_name]
                room_name, room_key = self._select_name(room_names_dict, available_keys, False)
                if room_name:
                    if current_room_name not in rooms_visited_state1:
                        rooms_visited_state1[current_room_name] = (room_name, [])
                    rooms_visited_state1[current_room_name][1].append(i)

        rooms_visited_state2 = {}
        for i, step in enumerate(trajectory2):
            current_room_name = step['room_name']
            if current_room_name in self.room_names[state2]:
                room_names_dict = self.room_names[state2][current_room_name]
                room_name, room_key = self._select_name(room_names_dict, available_keys, False)
                if room_name:
                    if current_room_name not in rooms_visited_state2:
                        rooms_visited_state2[current_room_name] = (room_name, [])
                    rooms_visited_state2[current_room_name][1].append(i)
        
        only_state1_rooms = list(set(rooms_visited_state1.keys()) - set(rooms_visited_state2.keys()))
        only_state2_rooms = list(set(rooms_visited_state2.keys()) - set(rooms_visited_state1.keys()))
        union_rooms = list(set(rooms_visited_state1.keys()) | set(rooms_visited_state2.keys()))
        inter_rooms = list(set(rooms_visited_state1.keys()) & set(rooms_visited_state2.keys()))

        questions.extend(self._generate_only_state1_visible_questions(state1,
                                                                      state2,
                                                                      union_rooms,
                                                                      inter_rooms,
                                                                      only_state1_rooms,
                                                                      only_state2_rooms,
                                                                      rooms_visited_state1,
                                                                      rooms_visited_state2))
        questions.extend(self._generate_only_state2_visible_questions(state1,
                                                                      state2,
                                                                      union_rooms,
                                                                      inter_rooms,
                                                                      only_state1_rooms,
                                                                      only_state2_rooms,
                                                                      rooms_visited_state1,
                                                                      rooms_visited_state2))
        questions.extend(self._generate_rooms_visited_both_states_questions(state1,
                                                                           state2,
                                                                           union_rooms,
                                                                           inter_rooms,
                                                                           only_state1_rooms,
                                                                           only_state2_rooms,
                                                                           rooms_visited_state1,
                                                                           rooms_visited_state2))
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions


    def _generate_nth_mth_room_type_questions(self, state1: str, state2: str, room_combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        type_question_template = self.templates['nth_mth_visit_room_type']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        len_video1 = len(trajectory1)
        video_num1 = self.get_video_num(state1)
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        len_video2 = len(trajectory2)
        video_num2 = self.get_video_num(state2)

        same_type_combinations = [combo for combo in room_combinations if combo['room1_type'] == combo['room2_type']]
        diff_type_combinations = [combo for combo in room_combinations if combo['room1_type'] != combo['room2_type']]
        
        target_count = min(len(same_type_combinations), len(diff_type_combinations))
        if target_count > 0:
            max_questions = min(target_count, 3)
            selected_combinations = random.sample(same_type_combinations, max_questions) + random.sample(diff_type_combinations, max_questions)
            random.shuffle(selected_combinations)
            
            for combo in selected_combinations:
                if combo['visit1_order'] == 1:
                    visit1_ordinal = "1st"
                elif combo['visit1_order'] == 2:
                    visit1_ordinal = "2nd"
                elif combo['visit1_order'] == 3:
                    visit1_ordinal = "3rd"
                else:
                    visit1_ordinal = f"{combo['visit1_order']}th"
                    
                if combo['visit2_order'] == 1:
                    visit2_ordinal = "1st"
                elif combo['visit2_order'] == 2:
                    visit2_ordinal = "2nd"
                elif combo['visit2_order'] == 3:
                    visit2_ordinal = "3rd"
                else:
                    visit2_ordinal = f"{combo['visit2_order']}th"
                
                room1_type = combo['room1_type']
                room2_type = combo['room2_type']
                all_types = list(self.room_types.keys())
                
                if room1_type == room2_type:
                    other_types = [t for t in all_types if t != room1_type]
                    correct_choice = f"Yes, they are both {room1_type}"
                    
                    yes_options = [correct_choice]
                    selected_other_type = random.choice(other_types)
                    yes_options.append(f"Yes, they are both {selected_other_type}")
                    
                    no_options = []
                    used_combinations = set()
                    
                    while len(no_options) < 2 and len(used_combinations) < len(other_types) * (len(other_types) - 1):
                        selected_other_types = random.sample(other_types, 2)
                        combination = tuple(selected_other_types)
                        
                        if combination not in used_combinations:
                            used_combinations.add(combination)
                            no_options.append(f"No, the formar room is {selected_other_types[0]} and the latter room is {selected_other_types[1]}")
                    
                    while len(no_options) < 2:
                        selected_type = random.choice(other_types)
                        option_text = f"No, the formar room is {selected_type} and the latter room is {room1_type}"
                        if option_text not in no_options:
                            no_options.append(option_text)
                        else:
                            selected_type = random.choice([t for t in other_types if t != selected_type])
                            option_text = f"No, the formar room is {selected_type} and the latter room is {room1_type}"
                            if option_text not in no_options:
                                no_options.append(option_text)
                    
                    choices = yes_options + no_options
                else:
                    other_types = [t for t in all_types if t != room1_type and t != room2_type]
                    
                    yes_options = [
                        f"Yes, they are both {room1_type}",
                        f"Yes, they are both {room2_type}"
                    ]
                    
                    correct_choice = f"No, the formar room is {room1_type} and the latter room is {room2_type}"
                    
                    no_options = [correct_choice]
                    
                    if len(other_types) >= 2:
                        selected_other_types = random.sample(other_types, 2)
                        if random.random() > 0.5:
                            no_options.append(f"No, the formar room is {room1_type} and the latter room is {selected_other_types[0]}")
                        else:
                            no_options.append(f"No, the formar room is {selected_other_types[1]} and the latter room is {room2_type}")
                    elif len(other_types) == 1:
                        selected_other_type = other_types[0]
                        if random.random() > 0.5:
                            no_options.append(f"No, the formar room is {room1_type} and the latter room is {selected_other_type}")
                        else:
                            no_options.append(f"No, the formar room is {selected_other_type} and the latter room is {room2_type}")

                    choices = yes_options + no_options

                is_hallucination = random.random() < 1/3
                if is_hallucination:
                    choices.remove(correct_choice)
                    choices.append(self.no_valid_option)
                    correct_choice = self.no_valid_option
                else:
                    candidate_remove_ids = [i for i, choice in enumerate(choices) if choice != correct_choice]
                    remove_id = random.choice(candidate_remove_ids)
                    choices.remove(choices[remove_id])
                    choices.append(self.no_valid_option)
                random.shuffle(choices)
                correct_answer = choices.index(correct_choice)

                answer_source = []              
                # answer_source = [{'type': 'text', 'content': f'The {video_num1} video has {len_video1} frames and the {video_num2} video has {len_video2} frames.'}]
                start_id, end_id = combo['room1_frames']
                clip1 = list(range(start_id, end_id + 1))
                # clip1_data = self.get_one_clip_data(clip1, state1)
                # answer_source.append({'type': 'text', 'content': f'In {video_num1} video, the {visit1_ordinal} visited room appears from frame {start_id} to frame {end_id}. This room is a {room1_type}.'})
                # answer_source.append({'type': 'video', 'content': clip1_data, 'state': state1})
                start_id, end_id = combo['room2_frames']
                clip2 = list((range(start_id, end_id + 1)))
                # clip2_data = self.get_one_clip_data(clip2, state2)
                # answer_source.append({'type': 'text', 'content': f'In {video_num2} video, the {visit2_ordinal} visited room appears from frame {start_id} to frame {end_id}. This room is a {room2_type}.'})
                # answer_source.append({'type': 'video', 'content': clip2_data, 'state': state2})
                
                related_frames = {state1: clip1, state2: clip2}
                
                question_text = type_question_template.format(state1=video_num1, state2=video_num2, visit1_order=visit1_ordinal, visit2_order=visit2_ordinal)
                questions.append({
                    'question': question_text,
                    'choices': choices,
                    'correct_answer': correct_answer,
                    'question_type': 'single_choice',
                    'category': 'multi_states_agent_explore',
                    'subcategory': 'nth_mth_visit_room_type',
                    'capabilities': self.templates['nth_mth_visit_room_type']['capabilities'],
                    'answer_source': answer_source,
                    'related_frames': related_frames,
                    'hallucination': is_hallucination
                })

        return questions

    def _generate_nth_mth_room_shape_questions(self, state1: str, state2: str, room_combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        shape_question_template = self.templates['nth_mth_visit_room_shape']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        len_video1 = len(trajectory1)
        video_num1 = self.get_video_num(state1)
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        len_video2 = len(trajectory2)
        video_num2 = self.get_video_num(state2)

        same_shape_combinations = [combo for combo in room_combinations if combo['room1_shape'] == combo['room2_shape']]
        diff_shape_combinations = [combo for combo in room_combinations if combo['room1_shape'] != combo['room2_shape']]
        
        target_count = min(len(same_shape_combinations), len(diff_shape_combinations))
        if target_count > 0:
            max_questions = min(target_count, 3)
            selected_combinations = random.sample(same_shape_combinations, max_questions) + random.sample(diff_shape_combinations, max_questions)
            random.shuffle(selected_combinations)
            
            for combo in selected_combinations:
                if combo['visit1_order'] == 1:
                    visit1_ordinal = "1st"
                elif combo['visit1_order'] == 2:
                    visit1_ordinal = "2nd"
                elif combo['visit1_order'] == 3:
                    visit1_ordinal = "3rd"
                else:
                    visit1_ordinal = f"{combo['visit1_order']}th"
                    
                if combo['visit2_order'] == 1:
                    visit2_ordinal = "1st"
                elif combo['visit2_order'] == 2:
                    visit2_ordinal = "2nd"
                elif combo['visit2_order'] == 3:
                    visit2_ordinal = "3rd"
                else:
                    visit2_ordinal = f"{combo['visit2_order']}th"
                
                room1_shape = combo['room1_shape']
                room2_shape = combo['room2_shape']
                all_shapes = self.room_shape_options
                
                if room1_shape == room2_shape:
                    other_shapes = [s for s in all_shapes if s != room1_shape]
                    correct_choice = f"Yes, they are both {room1_shape}"
                    
                    yes_options = [correct_choice]
                    selected_other_shape = random.choice(other_shapes)
                    yes_options.append(f"Yes, they are both {selected_other_shape}")
                    
                    no_options = []
                    used_combinations = set()
                    
                    while len(no_options) < 2 and len(used_combinations) < len(other_shapes) * (len(other_shapes) - 1):
                        selected_other_shapes = random.sample(other_shapes, 2)
                        combination = tuple(selected_other_shapes)
                        
                        if combination not in used_combinations:
                            used_combinations.add(combination)
                            no_options.append(f"No, the formar room is {selected_other_shapes[0]} and the latter room is {selected_other_shapes[1]}")
                    
                    while len(no_options) < 2:
                        selected_shape = random.choice(other_shapes)
                        option_text = f"No, the formar room is {selected_shape} and the latter room is {room1_shape}"
                        if option_text not in no_options:
                            no_options.append(option_text)
                        else:
                            selected_shape = random.choice([s for s in other_shapes if s != selected_shape])
                            option_text = f"No, the formar room is {selected_shape} and the latter room is {room1_shape}"
                            if option_text not in no_options:
                                no_options.append(option_text)
                    
                    choices = yes_options + no_options
                else:
                    other_shapes = [s for s in all_shapes if s != room1_shape and s != room2_shape]
                    
                    yes_options = [
                        f"Yes, they are both {room1_shape}",
                        f"Yes, they are both {room2_shape}"
                    ]
                    
                    correct_choice = f"No, the formar room is {room1_shape} and the latter room is {room2_shape}"
                    
                    no_options = [correct_choice]
                    
                    if len(other_shapes) >= 2:
                        selected_other_shapes = random.sample(other_shapes, 2)
                        if random.random() > 0.5:
                            no_options.append(f"No, the formar room is {room1_shape} and the latter room is {selected_other_shapes[0]}")
                        else:
                            no_options.append(f"No, the formar room is {selected_other_shapes[1]} and the latter room is {room2_shape}")
                    elif len(other_shapes) == 1:
                        selected_other_shape = other_shapes[0]
                        if random.random() > 0.5:
                            no_options.append(f"No, the formar room is {room1_shape} and the latter room is {selected_other_shape}")
                        else:
                            no_options.append(f"No, the formar room is {selected_other_shape} and the latter room is {room2_shape}")

                    choices = yes_options + no_options
                
                is_hallucination = random.random() < 1/3
                if is_hallucination:
                    choices.remove(correct_choice)
                    choices.append(self.no_valid_option)
                    correct_choice = self.no_valid_option
                else:
                    candidate_remove_ids = [i for i, choice in enumerate(choices) if choice != correct_choice]
                    remove_id = random.choice(candidate_remove_ids)
                    choices.remove(choices[remove_id])
                    choices.append(self.no_valid_option)
                random.shuffle(choices)
                correct_answer = choices.index(correct_choice)
                                
                answer_source = []
                # answer_source = [{'type': 'text', 'content': f'The {video_num1} video has {len_video1} frames and the {video_num2} video has {len_video2} frames.'}]
                start_id, end_id = combo['room1_frames']
                clip1 = list(range(start_id, end_id + 1))
                # clip1_data = self.get_one_clip_data(clip1, state1)
                # answer_source.append({'type': 'text', 'content': f'In {video_num1} video, the {visit1_ordinal} visited room appears from frame {start_id} to frame {end_id}. This room is {room1_shape}.'})
                # answer_source.append({'type': 'video', 'content': clip1_data, 'state': state1})
                start_id, end_id = combo['room2_frames']
                clip2 = list((range(start_id, end_id + 1)))
                # clip2_data = self.get_one_clip_data(clip2, state2)
                # answer_source.append({'type': 'text', 'content': f'In {video_num2} video, the {visit2_ordinal} visited room appears from frame {start_id} to frame {end_id}. This room is {room2_shape}.'})
                # answer_source.append({'type': 'video', 'content': clip2_data, 'state': state2})
                
                related_frames = {state1: clip1, state2: clip2}
                
                question_text = shape_question_template.format(state1=video_num1, state2=video_num2, visit1_order=visit1_ordinal, visit2_order=visit2_ordinal)
                questions.append({
                    'question': question_text,
                    'choices': choices,
                    'correct_answer': correct_answer,
                    'question_type': 'single_choice',
                    'category': 'multi_states_agent_explore',
                    'subcategory': 'nth_mth_visit_room_shape',
                    'capabilities': self.templates['nth_mth_visit_room_shape']['capabilities'],
                    'answer_source': answer_source,
                    'related_frames': related_frames,
                    'hallucination': is_hallucination
                })

        return questions

    def _generate_nth_mth_room_size_questions(self, state1: str, state2: str, room_combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        size_question_template = self.templates['nth_mth_visit_room_size']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        len_video1 = len(trajectory1)
        video_num1 = self.get_video_num(state1)
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        len_video2 = len(trajectory2)
        video_num2 = self.get_video_num(state2)

        diff_size_combinations = [combo for combo in room_combinations if abs(combo['room1_area'] - combo['room2_area'])/min(combo['room1_area'], combo['room2_area']) >= 0.1]
        
        max_questions = min(3, len(diff_size_combinations))
        selected_combinations = random.sample(diff_size_combinations, max_questions)
        
        for combo in selected_combinations:
            if combo['visit1_order'] == 1:
                visit1_ordinal = "1st"
            elif combo['visit1_order'] == 2:
                visit1_ordinal = "2nd"
            elif combo['visit1_order'] == 3:
                visit1_ordinal = "3rd"
            else:
                visit1_ordinal = f"{combo['visit1_order']}th"
                
            if combo['visit2_order'] == 1:
                visit2_ordinal = "1st"
            elif combo['visit2_order'] == 2:
                visit2_ordinal = "2nd"
            elif combo['visit2_order'] == 3:
                visit2_ordinal = "3rd"
            else:
                visit2_ordinal = f"{combo['visit2_order']}th"
            
            room1_area = combo['room1_area']
            room2_area = combo['room2_area']
            
            if room1_area > room2_area:
                percentage_diff = ((room1_area - room2_area) / room2_area) * 100
                lower_bound = max(0, percentage_diff - 5)
                upper_bound = percentage_diff + 5
                percentage_range = f"{lower_bound:.0f}-{upper_bound:.0f}%"
                correct_choice = f"The formar one is bigger, and it's about {percentage_range} larger than the latter one"
                
                wrong_percentage_range1 = f"{lower_bound+10:.0f}-{upper_bound+10:.0f}%"
                wrong_percentage_range2 = f"{max(0, lower_bound-10):.0f}-{max(0, upper_bound-10):.0f}%"
                
                choices = [
                    correct_choice,
                    f"The latter one is bigger, and it's about {percentage_range} larger than the formar one",
                    f"The formar one is bigger, and it's about {wrong_percentage_range1} larger than the latter one",
                    f"The latter one is bigger, and it's about {wrong_percentage_range2} larger than the formar one"
                ]
            else:
                percentage_diff = ((room2_area - room1_area) / room1_area) * 100
                lower_bound = max(0, percentage_diff - 5)
                upper_bound = percentage_diff + 5
                percentage_range = f"{lower_bound:.0f}-{upper_bound:.0f}%"
                correct_choice = f"The latter one is bigger, and it's about {percentage_range} larger than the formar one"
                
                wrong_percentage_range1 = f"{lower_bound+10:.0f}-{upper_bound+10:.0f}%"
                wrong_percentage_range2 = f"{max(0, lower_bound-10):.0f}-{max(0, upper_bound-10):.0f}%"
                
                choices = [
                    correct_choice,
                    f"The formar one is bigger, and it's about {percentage_range} larger than the latter one",
                    f"The latter one is bigger, and it's about {wrong_percentage_range1} larger than the formar one",
                    f"The formar one is bigger, and it's about {wrong_percentage_range2} larger than the latter one"
                ]
            
            is_hallucination = random.random() < 1/3
            if is_hallucination:
                choices.remove(correct_choice)
                choices.append(self.no_valid_option)
                correct_choice = self.no_valid_option
            else:
                candidate_remove_ids = [i for i, choice in enumerate(choices) if choice != correct_choice]
                remove_id = random.choice(candidate_remove_ids)
                choices.remove(choices[remove_id])
                choices.append(self.no_valid_option)
            random.shuffle(choices)
            correct_answer = choices.index(correct_choice)
                            
            answer_source = []
            # answer_source = [{'type': 'text', 'content': f'The {video_num1} video has {len_video1} frames and the {video_num2} video has {len_video2} frames.'}]
            
            start_id, end_id = combo['room1_frames']
            clip1 = list((range(start_id, end_id + 1)))
            # selected_obj_ids = self.select_proper_objects_in_frames(state1, start_id, end_id)
            # clip1_data = self.get_one_clip_data(clip1, state1, objs = selected_obj_ids, rooms = True, room_area = True)
            # answer_source.append({'type': 'text', 'content': f'In {video_num1} video, the {visit1_ordinal} visited room appears from frame {start_id} to frame {end_id}. This room is a {combo["room1_type"]} with an area of {room1_area:.2f} m².'})
            # answer_source.append({'type': 'video', 'content': clip1_data, 'state': state1})

            start_id, end_id = combo['room2_frames']
            clip2 = list((range(start_id, end_id + 1)))
            # selected_obj_ids = self.select_proper_objects_in_frames(state2, start_id, end_id)
            # clip2_data = self.get_one_clip_data(clip2, state2, objs = selected_obj_ids, rooms = True, room_area = True)
            # answer_source.append({'type': 'text', 'content': f'In {video_num2} video, the {visit2_ordinal} visited room appears from frame {start_id} to frame {end_id}. This room is a {combo["room2_type"]} with an area of {room2_area:.2f} m².'})
            # answer_source.append({'type': 'video', 'content': clip2_data, 'state': state2})
            
            related_frames = {state1: clip1, state2: clip2}
            
            question_text = size_question_template.format(state1=video_num1, state2=video_num2, visit1_order=visit1_ordinal, visit2_order=visit2_ordinal)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': 'single_choice',
                'category': 'multi_states_agent_explore',
                'subcategory': 'nth_mth_room_area_comparison',
                'capabilities': self.templates['nth_mth_visit_room_size']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })

        return questions

    def _generate_nth_mth_room_same_questions(self, state1: str, state2: str, room_combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        same_question_template = self.templates['nth_mth_visit_room_same']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        len_video1 = len(trajectory1)
        video_num1 = self.get_video_num(state1)
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        len_video2 = len(trajectory2)
        video_num2 = self.get_video_num(state2)

        same_room_combinations = [combo for combo in room_combinations if combo['room1_id'] == combo['room2_id']]
        diff_room_combinations = [combo for combo in room_combinations if combo['room1_id'] != combo['room2_id']]
        
        target_count = min(len(same_room_combinations), len(diff_room_combinations))
        if target_count > 0:
            max_questions = min(target_count, 3)
            selected_combinations = random.sample(same_room_combinations, max_questions) + random.sample(diff_room_combinations, max_questions)
            random.shuffle(selected_combinations)
            
            for combo in selected_combinations:
                if combo['visit1_order'] == 1:
                    visit1_ordinal = "1st"
                elif combo['visit1_order'] == 2:
                    visit1_ordinal = "2nd"
                elif combo['visit1_order'] == 3:
                    visit1_ordinal = "3rd"
                else:
                    visit1_ordinal = f"{combo['visit1_order']}th"
                    
                if combo['visit2_order'] == 1:
                    visit2_ordinal = "1st"
                elif combo['visit2_order'] == 2:
                    visit2_ordinal = "2nd"
                elif combo['visit2_order'] == 3:
                    visit2_ordinal = "3rd"
                else:
                    visit2_ordinal = f"{combo['visit2_order']}th"
                
                room1 = self.all_rooms[combo['room1_id']]
                room2 = self.all_rooms[combo['room2_id']]
                room1_type = room1['room_type']
                room2_type = room2['room_type']
                choices = ["Yes, they are the same room", "No, they are different rooms"]
                
                if combo['room1_id'] == combo['room2_id']:
                    correct_choice = "Yes, they are the same room"
                    correct_answer = choices.index(correct_choice)
                else:
                    correct_choice = "No, they are different rooms"
                    correct_answer = choices.index(correct_choice)
                
                answer_source = []
                # answer_source = [{'type': 'text', 'content': f'The {video_num1} video has {len_video1} frames and the {video_num2} video has {len_video2} frames.'}]
                
                start_id, end_id = combo['room1_frames']
                clip1 = list(range(start_id, end_id + 1))
                # clip1_data = self.get_one_clip_data(clip1, state1)
                # answer_source.append({'type': 'text', 'content': f'In {video_num1} video, the {visit1_ordinal} visited room appears from frame {start_id} to frame {end_id}. This room is a {room1_type} with an area of {room1["area"]:.2f} m².'})
                # answer_source.append({'type': 'video', 'content': clip1_data, 'state': state1})
                
                start_id, end_id = combo['room2_frames']
                clip2 = list(range(start_id, end_id + 1))
                # clip2_data = self.get_one_clip_data(clip2, state2)
                # answer_source.append({'type': 'text', 'content': f'In {video_num2} video, the {visit2_ordinal} visited room appears from frame {start_id} to frame {end_id}. This room is a {room2_type} with an area of {room2["area"]:.2f} m².'})
                # answer_source.append({'type': 'video', 'content': clip2_data, 'state': state2})

                related_frames = {state1: set(), state2: set()}
                for frame in clip1:
                    related_frames[state1].add(frame)
                for frame in clip2:
                    related_frames[state2].add(frame)
                related_frames[state1] = sorted(list(related_frames[state1]))
                related_frames[state2] = sorted(list(related_frames[state2]))
                
                question_text = same_question_template.format(state1=video_num1, state2=video_num2, visit1_order=visit1_ordinal, visit2_order=visit2_ordinal)
                questions.append({
                    'question': question_text,
                    'choices': choices,
                    'correct_answer': correct_answer,
                    'question_type': 'single_choice',
                    'category': 'multi_states_agent_explore',
                    'subcategory': 'nth_mth_room_identity_comparison',
                    'capabilities': self.templates['nth_mth_visit_room_same']['capabilities'],
                    'answer_source': answer_source,
                    'related_frames': related_frames,
                    'hallucination': False
                })        
        
        return questions

    def _generate_nth_mth_room_comparison_questions(self, state1: str, state2: str) -> List[Dict[str, Any]]:
        questions = []
        
        trajectory1 = self.scene_data[state1]['agent_trajectory']
        if len(trajectory1) < 2:
            return questions
        state1_visited_room_order = []
        last_room_id = None
        start_id = 0
        for i, point in enumerate(trajectory1):
            if 'room_name' not in point or 'position' not in point:
                continue
                
            current_room_id = point['room_name']

            if i == len(trajectory1) - 1:
                state1_visited_room_order.append((current_room_id, (start_id, i)))
            else:
                if current_room_id != last_room_id:
                    if last_room_id is not None:
                        state1_visited_room_order.append((last_room_id, (start_id, i - 1)))
                    last_room_id = current_room_id
                    start_id = i

        trajectory2 = self.scene_data[state2]['agent_trajectory']
        if len(trajectory2) < 2:
            return questions
        state2_visited_room_order = []
        last_room_id = None
        start_id = 0
        for i, point in enumerate(trajectory2):
            if 'room_name' not in point or 'position' not in point:
                continue
                
            current_room_id = point['room_name']

            if i == len(trajectory2) - 1:
                state2_visited_room_order.append((current_room_id, (start_id, i)))
            else:
                if current_room_id != last_room_id:
                    if last_room_id is not None:
                        state2_visited_room_order.append((last_room_id, (start_id, i - 1)))
                    last_room_id = current_room_id
                    start_id = i

        room_combinations = []
        for i, (room1_id, (start1_id, end1_id)) in enumerate(state1_visited_room_order):
            for j, (room2_id, (start2_id, end2_id)) in enumerate(state2_visited_room_order):
                visit1_order = i + 1
                visit2_order = j + 1
                
                room1 = self.all_rooms[room1_id]
                room2 = self.all_rooms[room2_id]
                room1_type = room1['room_type']
                room2_type = room2['room_type']
                room1_area = room1['area']
                room2_area = room2['area']
                room1_shape = room1['shape']
                room2_shape = room2['shape']
                
                room_combinations.append({
                    'room1_id': room1_id,
                    'room2_id': room2_id,
                    'room1_frames': (start1_id, end1_id),
                    'room2_frames': (start2_id, end2_id),
                    'visit1_order': visit1_order,
                    'visit2_order': visit2_order,
                    'room1_type': room1_type,
                    'room2_type': room2_type,
                    'room1_area': room1_area,
                    'room2_area': room2_area,
                    'room1_shape': room1_shape,
                    'room2_shape': room2_shape,
                })
        
        questions.extend(self._generate_nth_mth_room_type_questions(state1, state2, room_combinations))
        questions.extend(self._generate_nth_mth_room_shape_questions(state1, state2, room_combinations))
        questions.extend(self._generate_nth_mth_room_size_questions(state1, state2, room_combinations))
        questions.extend(self._generate_nth_mth_room_same_questions(state1, state2, room_combinations))

        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions