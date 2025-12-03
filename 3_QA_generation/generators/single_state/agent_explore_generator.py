import os
import json
import copy
from typing import List, Dict, Any, Tuple
import random
import inspect
from turtle import left
from typing import Any, Dict, List

from ..base_generator import BaseGenerator

class AgentExploreGenerator(BaseGenerator):
    
    def __init__(self, scene_path: str):
        super().__init__(scene_path)

        template_path = os.path.join(os.path.dirname(__file__), '../../templates/single_state/agent_explore_template.json')
        with open(template_path, 'r', encoding='utf-8') as f:
            self.templates = json.load(f)['agent_explore']
        
    def generate_questions(self) -> List[Dict[str, Any]]:
        questions = []
        
        for state_key in self.scene_data.keys():
            if state_key in ['room_static', 'object_mapping']:
                continue
            
            questions.extend(self._generate_room_visit_time_questions(state_key))
            questions.extend(self._generate_room_visit_count_questions(state_key))
            questions.extend(self._generate_nth_visit_room_questions(state_key))
            questions.extend(self._generate_room_successors_questions(state_key))
            questions.extend(self._generate_object_observation_time_questions(state_key))
            questions.extend(self._generate_object_appearance_time_questions(state_key))
        
        return questions

    def _generate_room_visit_shortest_time_questions(self, state_key: str, candidate_rooms: List[tuple], qa_num: int) -> List[Dict[str, Any]]:
        question_template = self.templates['room_shortest_time']['question_template']
        questions = []
        
        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)
        
        used_rooms = set()
        num = 0
        while True:
            available_candidates = [room for room in candidate_rooms if room[1] not in used_rooms]
            
            if len(available_candidates) < 4:
                break
            
            num_choices = min(random.randint(4, 6), len(available_candidates))
            selected_candidates = random.sample(available_candidates, num_choices)
            
            selected_candidates.sort(key=lambda x: len(x[2]))
            least_observed_room = selected_candidates[0]
            
            choices = [room[1] for room in selected_candidates]
            random.shuffle(choices)
            correct_answer = choices.index(least_observed_room[1])
            
            related_frames = {state_key: set()}
            
            answer_source = []
            answer_source.append({'type': 'text', 'content': f'There are {len_video} frames in the {video_num} video.'})
            for room in selected_candidates:
                room_id, room_name, _ = room
                room_frames = self.get_room_visit_frames(state_key, room_id)
                room_clips = self.find_continous_clips(room_frames)
                answer_source.append({'type': 'text', 'content': f'{room_name} was observed below:'})
                for i, clip in enumerate(room_clips):
                    start_frame = clip[0]
                    end_frame = clip[-1]
                    if i != len(room_clips) - 1:
                        answer_source.append({'type': 'text', 'content': f'frame {start_frame} to {end_frame}, '})
                    else:
                        answer_source.append({'type': 'text', 'content': f'frame {start_frame} to {end_frame}.'})
                    for frame in clip:
                        related_frames[state_key].add(frame)
            related_frames[state_key] = sorted(list(related_frames[state_key]))
            
            question_text = question_template.format(state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['room_shortest_time']['question_type'],
                'category': 'agent_explore',
                'subcategory': 'room_shortest_time',
                'capabilities': self.templates['room_shortest_time']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })
            
            for room in selected_candidates:
                used_rooms.add(room[1])

            num += 1
            if num > qa_num:
                break

        return questions

    def _generate_room_visit_longest_time_questions(self, state_key: str, candidate_rooms: List[tuple], qa_num: int) -> List[Dict[str, Any]]:
        question_template = self.templates['room_longest_time']['question_template']
        questions = []

        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)

        used_rooms = set()
        num = 0
        while True:
            available_candidates = [room for room in candidate_rooms if room[1] not in used_rooms]
            
            if len(available_candidates) < 4:
                break
            
            num_choices = min(random.randint(4, 6), len(available_candidates))
            selected_candidates = random.sample(available_candidates, num_choices)
            
            selected_candidates.sort(key=lambda x: len(x[2]))
            most_observed_room = selected_candidates[-1]
            
            choices = [room[1] for room in selected_candidates]
            random.shuffle(choices)
            correct_answer = choices.index(most_observed_room[1])

            related_frames = {state_key: set()}
            
            answer_source = []
            answer_source.append({'type': 'text', 'content': f'There are {len_video} frames in the {video_num} video.'})
            for room in selected_candidates:
                room_id, room_name, _ = room
                room_frames = self.get_room_visit_frames(state_key, room_id)
                room_clips = self.find_continous_clips(room_frames)
                answer_source.append({'type': 'text', 'content': f'{room_name} was observed below:'})
                for i, clip in enumerate(room_clips):
                    start_frame = clip[0]
                    end_frame = clip[-1]
                    if i != len(room_clips) - 1:
                        answer_source.append({'type': 'text', 'content': f'frame {start_frame} to {end_frame}, '})
                    else:
                        answer_source.append({'type': 'text', 'content': f'frame {start_frame} to {end_frame}.'})
                    for frame in clip:
                        related_frames[state_key].add(frame)
            related_frames[state_key] = sorted(list(related_frames[state_key]))
            
            question_text = question_template.format(state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['room_longest_time']['question_type'],
                'category': 'agent_explore',
                'subcategory': 'room_longest_time',
                'capabilities': self.templates['room_longest_time']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })
            
            for room in selected_candidates:
                used_rooms.add(room[1])
            
            num += 1
            if num > qa_num:
                break
        

        return questions

    def _generate_room_visit_time_questions(self, state_key: str) -> List[Dict[str, Any]]:
        available_keys = ['type', 'biggest_room', 'smallest_type', 'biggest_type', 'smallest_type', 'shape', 'unique_object']
        questions = []
        
        trajectory = self.scene_data[state_key]['agent_trajectory']

        rooms = self.scene_data['room_static']['room_static_details']
        available_rooms = {}
        for room in rooms:
            room_id = room['room_name']
            if room_id in self.room_names[state_key]:
                room_names = self.room_names[state_key][room_id]
                room_name, room_key = self._select_name(room_names, available_keys) ###
                if room_name:
                    available_rooms[room_id] = room_name
        
        room_observation_counts = {}
        for current_frame_id, step in enumerate(trajectory):
            current_room_name = step['room_name']
            if current_room_name in available_rooms:
                if current_room_name not in room_observation_counts:
                    room_observation_counts[current_room_name] = [current_frame_id]
                else:
                    room_observation_counts[current_room_name].append(current_frame_id)
        
        if len(room_observation_counts) < 2:
            return questions
            
        candidate_rooms = [(room_id, available_rooms[room_id], frames) 
                          for room_id, frames in room_observation_counts.items()]
        
        candidate_rooms.sort(key=lambda x: len(x[2]))

        questions.extend(self._generate_room_visit_shortest_time_questions(state_key, candidate_rooms, 3))
        
        questions.extend(self._generate_room_visit_longest_time_questions(state_key, candidate_rooms, 3))
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    

    def _generate_room_visit_count_questions(self, state_key: str) -> List[Dict[str, Any]]:
        available_keys = ['type', 'biggest_room', 'smallest_room', 'biggest_type', 'smallest_type', 'shape', 'unique_object']
        question_template = self.templates['room_visit_count']['question_template']
        questions = []
        
        trajectory = self.scene_data[state_key]['agent_trajectory']
        video_num = self.get_video_num(state_key)
        
        if len(trajectory) < 2:
            return questions
        
        room_visit_counts = {}
        current_room_id = None
        
        for point in trajectory:
            if 'room_name' not in point:
                continue
                
            room_id = point['room_name']
            
            if current_room_id != room_id:
                if room_id not in room_visit_counts:
                    room_visit_counts[room_id] = 0
                room_visit_counts[room_id] += 1
                current_room_id = room_id
        
        for room_id, visit_count in room_visit_counts.items():
            if visit_count == 0:
                continue
            if random.random() * len(room_visit_counts) > 5:
                continue
            
            room_name = None
            if room_id in self.room_names[state_key]:
                room_names = self.room_names[state_key][room_id]
                room_name, room_key = self._select_name(room_names, available_keys) ###
            
            if not room_name:
                continue
                
            num_choices = random.randint(4, 6)
            choices = set([visit_count])  # Include correct answer
            
            # Generate other plausible visit counts using a normal distribution
            mean = visit_count
            std_dev = max(1, visit_count // 2)  # Standard deviation based on visit_count
            
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
            original_correct_answer = visit_count
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

            related_frames = {state_key: set()}

            answer_source = []
            len_video = len(self.scene_data[state_key]['agent_trajectory'])
            room_frames = self.get_room_visit_frames(state_key, room_id)
            room_clips = self.find_continous_clips(room_frames)
            answer_source.append({'type': 'text', 'content': f'There are {len_video} frames in the {video_num} video and observer is located in {room_name} in the following video clip:'})
            for i, clip in enumerate(room_clips):
                start_frame = clip[0]
                end_frame = clip[-1]
                if i != len(room_clips) - 1:
                    answer_source.append({'type': 'text', 'content': f'frame {start_frame} to {end_frame}, '})
                else:
                    answer_source.append({'type': 'text', 'content': f'frame {start_frame} to {end_frame}.'})
                for frame in clip:
                    related_frames[state_key].add(frame)
            related_frames[state_key] = sorted(list(related_frames[state_key]))

            question_text = question_template.format(room=room_name, state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_choice,
                'question_type': self.templates['room_visit_count']['question_type'],
                'category': 'agent_explore',
                'subcategory': 'room_visit_count',
                'capabilities': self.templates['room_visit_count']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    

    def _generate_nth_visit_room_type_questions(self, state_key: str, room_id: str, start_id: int, end_id: int, visit_ordinal: str, room_order_text: str) -> List[Dict[str, Any]]:
        question_template = self.templates['nth_visit_room_type']['question_template']
        questions = []

        room = self.all_rooms[room_id]
        room_type = room['room_type']
        other_types = list(self.room_types.keys())
        random.shuffle(other_types)
        num_choice = random.randint(4,6)
        choices = [room_type]
        for t in other_types:
            if t not in choices and len(choices) < num_choice:
                choices.append(t)
        
        original_correct_answer = room_type
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
        correct_answer = choices.index(correct_answer_value)

        answer_source = []
        answer_source.append({'type': 'text', 'content': room_order_text})
        clip = list(range(start_id, end_id + 1))
        # clip_data = self.get_one_clip_data(clip, state_key)
        # answer_source = [
        #     {'type': 'text', 'content': f'The {visit_ordinal} visited room appears from frame {start_id} to frame {end_id}.'},
        #     {'type': 'video', 'content': clip_data, 'state': state_key}
        # ]

        related_frames = {state_key: clip}

        question_text = question_template.format(
            visit_order=visit_ordinal, 
            state=state_key
        )
        questions.append({
            'question': question_text,
            'choices': choices,
            'correct_answer': correct_answer,
            'question_type': self.templates['nth_visit_room_type']['question_type'],
            'category': 'agent_explore',
            'subcategory': 'nth_visit_room_type',
            'capabilities': self.templates['nth_visit_room_type']['capabilities'],
            'answer_source': answer_source,
            'related_frames': related_frames,
            'hallucination': is_hallucination
        })

        return questions

    def _generate_nth_visit_room_area_questions(self, state_key: str, room_id: str, start_id: int, end_id: int, visit_ordinal: str, room_order_text: str) -> List[Dict[str, Any]]:
        question_template = self.templates['nth_visit_room_area']['question_template']
        questions = []

        room = self.all_rooms[room_id]
        area = room['area']
        num_choices = random.randint(4, 6)
        range_float = random.random()
        area_range = [area * range_float, area * (1 + range_float)]
        choices, original_correct_choice_index = self.create_range_choices(area_range, area, 'm²', num_choices)

        if not choices:
            return []

        if random.random() < 1/3:
            is_hallucination = True
            correct_answer = self.no_valid_option
            choices.pop(original_correct_choice_index)
            choices.append(correct_answer)
        else:
            is_hallucination = False
            correct_answer = copy.copy(choices[original_correct_choice_index])
            candidate_remove_ids = [i for i, c in enumerate(choices) if i != original_correct_choice_index]
            if candidate_remove_ids:
                remove_id = random.choice(candidate_remove_ids)
                choices.pop(remove_id)
            choices.append(self.no_valid_option)

        random.shuffle(choices)
        correct_choice = choices.index(correct_answer)

        # if random.random() < 1/3:
        #     is_hallucination = True
        #     correct_answer_value = self.no_valid_option
        #     choices.pop(original_correct_choice_index)
        #     choices.append(correct_answer_value)
        # else:
        #     is_hallucination = False
        #     correct_answer_value = copy.copy(choices[original_correct_choice_index])
        #     num_current_choices = len(choices)
        #     all_indices = list(range(num_current_choices))
        #     candidate_remove_ids = [idx for idx in all_indices if idx != original_correct_choice_index]
        #     if candidate_remove_ids:
        #         remove_id = random.choice(candidate_remove_ids)
        #         choices.pop(remove_id)
        #     choices.append(self.no_valid_option)

        # random.shuffle(choices)
        # correct_choice = choices.index(correct_answer_value)

        answer_source = []
        answer_source.append({'type': 'text', 'content': room_order_text})
        clip = list(range(start_id, end_id + 1))
        related_frames = {state_key: clip}

        question_text = question_template.format(
            visit_order=visit_ordinal, 
            state=state_key
        )
        questions.append({
            'question': question_text,
            'choices': choices,
            'correct_answer': correct_choice,
            'question_type': self.templates['nth_visit_room_area']['question_type'],
            'category': 'agent_explore',
            'subcategory': 'nth_visit_room_area',
            'capabilities': self.templates['nth_visit_room_area']['capabilities'],
            'answer_source': answer_source,
            'related_frames': related_frames,
            'hallucination': is_hallucination
        })
        return questions

    def _generate_nth_visit_room_shape_questions(self, state_key: str, room_id: str, start_id: int, end_id: int, visit_ordinal: str, room_order_text: str) -> List[Dict[str, Any]]:
        question_template = self.templates['nth_visit_room_shape']['question_template']
        questions = []

        room = self.all_rooms[room_id]
        correct_answer = room['shape']
        num_choices = random.randint(4, 6)
        available_shapes = [shape for shape in self.room_shape_options if shape != correct_answer]
        if len(available_shapes) >= num_choices - 1:
            distractor_choices = random.sample(available_shapes, num_choices - 1)
        else:
            distractor_choices = available_shapes
        
        original_correct_answer = correct_answer
        choices = distractor_choices + [original_correct_answer]
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

        answer_source = []
        answer_source.append({'type': 'text', 'content': room_order_text})
        clip = list(range(start_id, end_id + 1))
        # clip_data = self.get_one_clip_data(clip, state_key)
        # answer_source = [
        #     {'type': 'text', 'content': f'The {visit_ordinal} visited room appears from frame {start_id} to frame {end_id}.'},
        #     {'type': 'video', 'content': clip_data, 'state': state_key}
        # ]

        related_frames = {state_key: clip}

        question_text = question_template.format(
            visit_order=visit_ordinal, 
            state=state_key
        )
        questions.append({
            'question': question_text,
            'choices': choices,
            'correct_answer': correct_choice,
            'question_type': self.templates['nth_visit_room_shape']['question_type'],
            'category': 'agent_explore',
            'subcategory': 'nth_visit_room_shape',
            'capabilities': self.templates['nth_visit_room_shape']['capabilities'],
            'answer_source': answer_source,
            'related_frames': related_frames,
            'hallucination': is_hallucination
        })
        return questions
    
    def _generate_nth_visit_room_objects_questions(self, state_key: str, room_id: str, start_id: int, end_id: int, visit_ordinal: str, room_order_text: str, object_names: Dict[str, Tuple[str, str]]) -> List[Dict[str, Any]]:
        question_template = self.templates['nth_visit_room_objects']['question_template']
        questions = []

        num_choice = random.randint(3, min(5, len(object_names)))
        selected_objs = random.sample(list(object_names.items()), num_choice)
        # import pdb; pdb.set_trace()

        all_possible_choices = []
        original_correct_answers = []
        obj_ids = []
        for (obj_id, (obj_name, obj_room_id)) in selected_objs:
            all_possible_choices.append(obj_name)
            if room_id == obj_room_id:
                original_correct_answers.append(obj_name)
                obj_ids.append((obj_id, 0))

        if not original_correct_answers:
            is_hallucination = True
            correct_answer_values = [self.no_valid_option]
            choices = all_possible_choices
            if self.no_valid_option not in choices:
                choices.append(self.no_valid_option)
        else:
            if len(original_correct_answers) <=2 and random.random() < 1/3:
                is_hallucination = True
                correct_answer_values = [self.no_valid_option]
                choices = [c for c in all_possible_choices if c not in original_correct_answers]
                if self.no_valid_option not in choices:
                    choices.append(self.no_valid_option)
            else:
                is_hallucination = False
                correct_answer_values = original_correct_answers
                choices = all_possible_choices
                if self.no_valid_option not in choices:
                    choices.append(self.no_valid_option)

        random.shuffle(choices)
        correct_choice = [choices.index(ans) for ans in correct_answer_values]

        answer_source = []
        answer_source.append({'type': 'text', 'content': room_order_text})
        clip = list(range(start_id, end_id + 1))
        related_frames = {state_key: clip}

        question_text = question_template.format(
            visit_order=visit_ordinal,
            state=state_key
        )
        questions.append({
            'question': question_text,
            'choices': choices,
            'correct_answer': correct_choice,
            'question_type': self.templates['nth_visit_room_objects']['question_type'],
            'category': 'agent_explore',
            'subcategory': 'nth_visit_room_objects',
            'capabilities': self.templates['nth_visit_room_objects']['capabilities'],
            'answer_source': answer_source,
            'related_frames': related_frames,
            'hallucination': is_hallucination
        })
        return questions

    def _generate_nth_visit_room_questions(self, state_key: str) -> List[Dict[str, Any]]:
        available_keys = ['type', 'in_receptacle', 'with_contents', 'attribute']

        questions = []
        
        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)
        
        if len(trajectory) < 2:
            return questions
        
        visited_room_order = []
        last_room_id = None
        start_id = 0
        
        for i, point in enumerate(trajectory):
            if 'room_name' not in point or 'position' not in point:
                continue
                
            current_room_id = point['room_name']

            if i == len(trajectory) - 1:
                visited_room_order.append((current_room_id, (start_id, i)))
            else:
                if current_room_id != last_room_id:
                    if last_room_id is not None:
                        visited_room_order.append((last_room_id, (start_id, i - 1)))
                    last_room_id = current_room_id
                    start_id = i

        room_order_text = f"There are {len_video} frames in the {video_num} video in total.\n"
        for i, (room_id, (start_id, end_id)) in enumerate(visited_room_order):
            visit_order = i + 1
            if visit_order == 1:
                visit_ordinal = "1st"
            elif visit_order == 2:
                visit_ordinal = "2nd"
            elif visit_order == 3:
                visit_ordinal = "3rd"
            else:
                visit_ordinal = f"{visit_order}th"
            room_order_text += f"{visit_ordinal} visited room: frame {start_id} to {end_id}\n"
        
        object_names = {}
        room_objects = self.room_objects[state_key]
        rooms = self.scene_data['room_static']['room_static_details']
        for room in rooms:
            if room['room_name'] not in room_objects:
                continue
            room_objs = room_objects[room['room_name']]
            for obj in room_objs:
                obj_names = self.object_names[state_key][obj['objectId']]
                obj_name, obj_key = self._select_name(obj_names, available_keys)
                if obj_name:
                    # if room['room_name'] not in object_names:
                    #     object_names[room['room_name']] = {}
                    object_names[obj['objectId']] = (obj_name, room['room_name'])

        for i, (room_id, (start_id, end_id)) in enumerate(visited_room_order):
            visit_order = i + 1
            if visit_order == 1:
                visit_ordinal = "1st"
            elif visit_order == 2:
                visit_ordinal = "2nd"
            elif visit_order == 3:
                visit_ordinal = "3rd"
            else:
                visit_ordinal = f"{visit_order}th"

            if random.random() * len(visited_room_order) < 5/2: # type
                questions.extend(self._generate_nth_visit_room_type_questions(state_key, room_id, start_id, end_id, visit_ordinal, room_order_text))

            if random.random() * len(visited_room_order) < 5/2: # area
                questions.extend(self._generate_nth_visit_room_area_questions(state_key, room_id, start_id, end_id, visit_ordinal, room_order_text))

            if random.random() * len(visited_room_order) < 5/2: # shape
                questions.extend(self._generate_nth_visit_room_shape_questions(state_key, room_id, start_id, end_id, visit_ordinal, room_order_text))

            if random.random() * len(visited_room_order) < 5/2 and room_id in room_objects and len(object_names) > 3: # object
                questions.extend(self._generate_nth_visit_room_objects_questions(state_key, room_id, start_id, end_id, visit_ordinal, room_order_text, object_names))
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    

    def _generate_room_successors_questions(self, state_key: str) -> List[Dict[str, Any]]:
        available_keys = ['type', 'biggest_room', 'smallest_type', 'biggest_type', 'smallest_type', 'shape', 'unique_object']
        question_template = self.templates['room_visit_order']['question_template']
        questions = []

        available_rooms = {}
        for room in self.scene_data['room_static']['room_static_details']:
            room_id = room['room_name']
            if room_id in self.room_names[state_key]:
                room_names = self.room_names[state_key][room_id]
                room_name, room_key = self._select_name(room_names, available_keys) ###
                if room_name:
                    available_rooms[room_id] = room_name

        if len(available_rooms) < 3:
            return questions

        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)
        
        visited_room_order = []
        last_room_id = None
        start_id = 0
        for i, point in enumerate(trajectory):
            if 'room_name' not in point or 'position' not in point:
                continue
            current_room_id = point['room_name']
            if i == len(trajectory) - 1:
                visited_room_order.append((current_room_id, (start_id, i)))
            else:
                if current_room_id != last_room_id:
                    if last_room_id is not None:
                        visited_room_order.append((last_room_id, (start_id, i - 1)))
                    last_room_id = current_room_id
                    start_id = i
            
        if len(visited_room_order) < 2:
            return questions

        room_successors = {}
        for i in range(len(visited_room_order) - 1):
            current_room_id = visited_room_order[i][0]
            next_room_id = visited_room_order[i+1][0]
            current_room_name = available_rooms.get(current_room_id)
            next_room_name = available_rooms.get(next_room_id)
            if current_room_name and next_room_name:
                if current_room_name not in room_successors.keys():
                    room_successors[current_room_id] = {}
                if next_room_name not in room_successors[current_room_id].keys():
                    current_room_frame = visited_room_order[i][1]
                    next_room_frame = visited_room_order[i+1][1]
                    room_successors[current_room_id][next_room_id] = (current_room_frame[0], next_room_frame[1])

        if not room_successors:
            return questions

        all_visited_room_ids = list(available_rooms.keys())

        num = 0
        for current_room_id, successors in room_successors.items():
            num_choices = min(max(random.randint(4, 6), len(successors)), len(available_rooms) - 1)
            tmp_correct_answers = []
            all_possible_choices = []

            for room_id, frames in successors.items():
                room_name = available_rooms[room_id]
                tmp_correct_answers.append((room_name, frames))
                all_possible_choices.append(room_name)

            random.shuffle(all_visited_room_ids)
            for room_id in all_visited_room_ids:
                room_name = available_rooms[room_id]
                if room_name not in all_possible_choices and len(all_possible_choices) < num_choices:
                    all_possible_choices.append(room_name)

            is_hallucination = False
            correct_answer_values = []
            choices = []

            if not tmp_correct_answers:
                # is_hallucination = True
                # correct_answer_values = [self.no_valid_option]
                # choices = all_possible_choices
                # if self.no_valid_option not in choices:
                #     choices.append(self.no_valid_option)
                continue
            else:
                if len(tmp_correct_answers) <= 2 and random.random() < 1/3:
                    is_hallucination = True
                    correct_answer_values = [self.no_valid_option]
                    correct_room_names = [ans[0] for ans in tmp_correct_answers]
                    choices = [c for c in all_possible_choices if c not in correct_room_names]
                    if self.no_valid_option not in choices:
                        choices.append(self.no_valid_option)
                else:
                    is_hallucination = False
                    correct_answer_values = [ans[0] for ans in tmp_correct_answers]
                    choices = all_possible_choices
                    if self.no_valid_option not in choices:
                        choices.append(self.no_valid_option)

            random.shuffle(choices)
            correct_answer = [choices.index(ans) for ans in correct_answer_values]

            answer_source = []
            # answer_source = [
            #     {'type': 'text', 'content': f"There are {len_video} frames in the {video_num} video."},
            #     {'type': 'text', 'content': "The following video clips show room visit order:"}
            # ]
            # for tmp_correct_answer in tmp_correct_answers:
            #     room_name, frames = tmp_correct_answer
            #     start_frame, end_frame = frames
            #     clip = list(range(start_frame, end_frame + 1))
            #     clip_data = self.get_one_clip_data(clip, state_key)
            #     answer_source.append({'type': 'text', 'content': f'The video clip is extracted from frames {start_frame} to {end_frame}.'})
            #     answer_source.append({'type': 'text', 'content': 'Then answer the question based on the objects placed above or inside it.'})
            #     answer_source.append({'type': 'video', 'content': clip_data, 'state': state_key})

            related_frames = {state_key: set()}
            for tmp_correct_answer in tmp_correct_answers:
                room_name, frames = tmp_correct_answer
                start_frame, end_frame = frames
                for frame_id in range(start_frame, end_frame + 1):
                    related_frames[state_key].add(frame_id)
            related_frames[state_key] = sorted(list(related_frames[state_key]))

            question_text = question_template.format(state=state_key, room=available_rooms[current_room_id])
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['room_visit_order']['question_type'],
                'category': 'agent_explore',
                'subcategory': 'room_visit_order',
                'capabilities': self.templates['room_visit_order']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })

            num += 1
            if num > 5:
                break

        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    

    def _generate_object_longest_observation_time_questions(self, state_key: str, candidate_objs: List[Tuple[str, str, List[int]]], qa_num: int) -> List[Dict[str, Any]]:
        question_template = self.templates['object_observation_longest']['question_template']
        questions = []

        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)

        used_objects = set()
        num = 0
        while True:
            available_candidates = [obj for obj in candidate_objs if obj[1] not in used_objects]
            
            if len(available_candidates) < 4:
                break
            
            num_choices = min(random.randint(4, 6), len(available_candidates))
            selected_candidates = random.sample(available_candidates, num_choices)
            
            selected_candidates.sort(key=lambda x: len(x[2]))
            most_observed_obj = selected_candidates[-1]
            
            random.shuffle(selected_candidates)
            choices = [obj[1] for obj in selected_candidates]
            correct_answer = choices.index(most_observed_obj[1])
            
            related_frames = {state_key: set()}
            answer_source = [{'type': 'text', 'content': f'There are {len_video} frames in the {video_num} video in total.'}]
            for obj in selected_candidates:
                obj_id, obj_name, frames = obj
                obj_clips = self.find_continous_clips(frames)
                answer_source.append({'type': 'text', 'content': f'{obj_name} was observed in below frames:'})
                for i, clip in enumerate(obj_clips):
                    start_frame = clip[0]
                    end_frame = clip[-1]
                    if i != len(obj_clips) - 1:
                        answer_source.append({'type': 'text', 'content': f'frame {start_frame} to {end_frame}, '})
                    else:
                        answer_source.append({'type': 'text', 'content': f'frame {start_frame} to {end_frame}.'})

                    for frame in clip:
                        related_frames[state_key].add(frame)
                # clip = min(obj_clips, key=lambda x: len(x))
                # clip_data = self.get_one_clip_data(clip, state_key, [obj_id])
                # answer_source.append({'type': 'text', 'content': f'Here is a video clip of {obj_name}:'})
                # answer_source.append({'type': 'video', 'content': clip_data, 'state': state_key})

            related_frames[state_key] = list(related_frames[state_key])

            question_text = question_template.format(state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['object_observation_longest']['question_type'],
                'category': 'agent_explore',
                'subcategory': 'object_observation_longest',
                'capabilities': self.templates['object_observation_longest']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })
            
            for obj in selected_candidates:
                used_objects.add(obj[1])

            num += 1
            if num > qa_num:
                break
        
        return questions

    def _generate_object_shortest_observation_time_questions(self, state_key: str, candidate_objs: List[Tuple[str, str, List[int]]], qa_num: int) -> List[Dict[str, Any]]:
        question_template = self.templates['object_observation_shortest']['question_template']
        questions = []

        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)

        used_objects = set()
        num = 0
        while True:
            available_candidates = [obj for obj in candidate_objs if obj[1] not in used_objects]
            
            if len(available_candidates) < 4:
                break
            
            num_choices = min(random.randint(4, 6), len(available_candidates))
            selected_candidates = random.sample(available_candidates, num_choices)
            
            selected_candidates.sort(key=lambda x: len(x[2]))
            least_observed_obj = selected_candidates[0]  # 观察次数最少的物体
            
            random.shuffle(selected_candidates)
            choices = [obj[1] for obj in selected_candidates]
            correct_answer = choices.index(least_observed_obj[1])
            
            related_frames = {state_key: set()}
            answer_source = [{'type': 'text', 'content': f'There are {len_video} frames in the {video_num} video in total.'}]
            for obj in selected_candidates:
                obj_id, obj_name, frames = obj
                obj_clips = self.find_continous_clips(frames)
                answer_source.append({'type': 'text', 'content': f'{obj_name} was observed in below frames:'})
                for i, clip in enumerate(obj_clips):
                    start_frame = clip[0]
                    end_frame = clip[-1]
                    if i != len(obj_clips) - 1:
                        answer_source.append({'type': 'text', 'content': f'frame {start_frame} to {end_frame}, '})
                    else:
                        answer_source.append({'type': 'text', 'content': f'frame {start_frame} to {end_frame}.'})
                    for frame in clip:
                        related_frames[state_key].add(frame)
                # clip = min(obj_clips, key=lambda x: len(x))
                # clip_data = self.get_one_clip_data(clip, state_key, [obj_id])
                # answer_source.append({'type': 'text', 'content': f'Here is a video clip of {obj_name}:'})
                # answer_source.append({'type': 'video', 'content': clip_data, 'state': state_key})
            related_frames[state_key] = list(related_frames[state_key])
            
            question_text = question_template.format(state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['object_observation_shortest']['question_type'],
                'category': 'agent_explore',
                'subcategory': 'object_observation_shortest',
                'capabilities': self.templates['object_observation_shortest']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })
            
            for obj in selected_candidates:
                used_objects.add(obj[1])

            num += 1
            if num > qa_num:
                break
        
        return questions

    def _generate_object_observation_time_questions(self, state_key: str) -> List[Dict[str, Any]]:
        available_keys = ['type', 'type_in_room', 'with_contents', 'in_receptacle', 'size', 'position', 'attribute']
        questions = []

        room_objects = self.room_objects[state_key]
        available_objects = {}
        for objects_in_room in room_objects.values():
            for obj in objects_in_room:
                obj_id = obj['objectId']
                if obj_id in self.object_names[state_key]:
                    obj_names = self.object_names[state_key][obj_id]
                    obj_name, obj_key = self._select_name(obj_names, available_keys) ###
                    if obj_name:
                        available_objects[obj_id] = obj_name

        trajectory = self.scene_data[state_key]['agent_trajectory']
        
        object_observation_counts = {}
        for i, step in enumerate(trajectory):
            visible_objects = step.get('visible_objects', [])
            for obj_id in visible_objects:
                if obj_id in available_objects:
                    if obj_id not in object_observation_counts:
                        object_observation_counts[obj_id] = []
                    object_observation_counts[obj_id].append(i)
        
        if len(object_observation_counts) < 4:
            return questions
            
        candidate_objs = [(obj_id, available_objects[obj_id], frames) 
                         for obj_id, frames in object_observation_counts.items()]
        
        # candidate_objs.sort(key=lambda x: x[2])
        
        questions.extend(self._generate_object_longest_observation_time_questions(state_key, candidate_objs, 3))
        
        questions.extend(self._generate_object_shortest_observation_time_questions(state_key, candidate_objs, 3))
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions


    def _generate_first_earliest_observation_time_questions(self, state_key: str, candidate_objs: List[Tuple[str, str, List[int]]], qa_num: int) -> List[Dict[str, Any]]:
        question_template = self.templates['object_first_appearance_earliest']['question_template']
        questions = []

        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)

        used_objects = set()
        num = 0
        while True:
            available_candidates = [obj for obj in candidate_objs if obj[1] not in used_objects]
            
            if len(available_candidates) < 4:
                break
            
            num_choices = min(random.randint(4, 6), len(available_candidates))
            selected_candidates = random.sample(available_candidates, num_choices)
            
            selected_candidates.sort(key=lambda x: x[2])
            earliest_time = selected_candidates[0][2]  # 最早的时间
            latest_time = selected_candidates[-1][2]  # 最晚的时间
            earliest_objects = [obj for obj in selected_candidates if obj[2] == earliest_time] # 所有具有最早时间的物体
            
            random.shuffle(selected_candidates)
            choices = [obj[1] for obj in selected_candidates]
            correct_answers = sorted([choices.index(obj[1]) for obj in earliest_objects])

            answer_source = []
            answer_source.append({'type': 'text', 'content': f'There are {len_video} frames in the {video_num} video in total.'})
            for obj in selected_candidates:
                obj_id, obj_name, time = obj
                answer_source.append({'type': 'text', 'content': f'{obj_name} first appears at frame {time}.'})
            obj_ids = [(obj[0], 0) for obj in selected_candidates]
            start_id = max(0, earliest_time - 5)
            end_id = min(len(trajectory) - 1, earliest_time + 5)
            clip = list(range(start_id, end_id))
            # clip_data = self.get_one_clip_data(clip, state_key, obj_ids)
            # answer_source.append({'type': 'text', 'content': f'Here is a video clip showing the first appearance times of objects in each option, spanning from frame {start_id} to frame {end_id}:'})
            # answer_source.append({'type': 'video', 'content': clip_data, 'state': state_key})

            related_frames = {state_key: clip}
            
            question_text = question_template.format(state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answers,
                'question_type': self.templates['object_first_appearance_earliest']['question_type'],
                'category': 'agent_explore',
                'subcategory': 'object_first_appearance_earliest',
                'capabilities': self.templates['object_first_appearance_earliest']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })
            
            for obj in selected_candidates:
                used_objects.add(obj[1])

            num += 1
            if num > qa_num:
                break

        return questions

    def _generate_first_latest_observation_time_questions(self, state_key: str, candidate_objs: List[Tuple[str, str, List[int]]], qa_num: int) -> List[Dict[str, Any]]:
        question_template = self.templates['object_first_appearance_latest']['question_template']
        questions = []

        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)

        used_objects = set()
        num = 0
        while True:
            available_candidates = [obj for obj in candidate_objs if obj[1] not in used_objects]
            
            if len(available_candidates) < 4:
                break
            
            num_choices = min(random.randint(4, 6), len(available_candidates))
            selected_candidates = random.sample(available_candidates, num_choices)
            
            selected_candidates.sort(key=lambda x: x[2])
            earliest_time = selected_candidates[0][2]  # 最早的时间
            latest_time = selected_candidates[-1][2]  # 最晚的时间
            latest_objects = [obj for obj in selected_candidates if obj[2] == latest_time]
            
            # 生成选项
            random.shuffle(selected_candidates)
            choices = [obj[1] for obj in selected_candidates]
            correct_answers = sorted([choices.index(obj[1]) for obj in latest_objects])

            answer_source = []
            answer_source.append({'type': 'text', 'content': f'There are {len_video} frames in the {video_num} video in total.'})
            for obj in selected_candidates:
                obj_id, obj_name, time = obj
                answer_source.append({'type': 'text', 'content': f'{obj_name} first appears at frame {time}.'})
            obj_ids = [(obj[0], 0) for obj in selected_candidates]
            start_id = max(0, earliest_time - 5)
            end_id = min(len(trajectory) - 1, earliest_time + 5)
            clip = list(range(start_id, end_id))
            # clip_data = self.get_one_clip_data(clip, state_key, obj_ids)
            # answer_source.append({'type': 'text', 'content': f'Here is a video clip showing the first appearance times of objects in each option, spanning from frame {start_id} to frame {end_id}:'})
            # answer_source.append({'type': 'video', 'content': clip_data, 'state': state_key})

            related_frames = {state_key: clip}
            
            question_text = question_template.format(state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answers,
                'question_type': self.templates['object_first_appearance_latest']['question_type'],
                'category': 'agent_explore',
                'subcategory': 'object_first_appearance_latest',
                'capabilities': self.templates['object_first_appearance_latest']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })
            
            for obj in selected_candidates:
                used_objects.add(obj[1])

            num += 1
            if num > qa_num:
                break
        
        return questions

    def _generate_last_earliest_observation_time_questions(self, state_key: str, candidate_objs: List[Tuple[str, str, List[int]]], qa_num: int) -> List[Dict[str, Any]]:
        question_template = self.templates['object_last_appearance_earliest']['question_template']
        questions = []

        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)

        used_objects = set()
        num = 0
        while True:
            available_candidates = [obj for obj in candidate_objs if obj[1] not in used_objects]
            
            if len(available_candidates) < 4:
                break
            
            num_choices = min(random.randint(4, 6), len(available_candidates))
            selected_candidates = random.sample(available_candidates, num_choices)
            
            # 按最后一次出现时间排序选中的候选物体
            selected_candidates.sort(key=lambda x: x[2])
            earliest_time = selected_candidates[0][2]  # 最早的时间
            latest_time = selected_candidates[-1][2]  # 最晚的时间
            earliest_objects = [obj for obj in selected_candidates if obj[2] == earliest_time]
            
            # 生成选项
            random.shuffle(selected_candidates)
            choices = [obj[1] for obj in selected_candidates]
            correct_answers = sorted([choices.index(obj[1]) for obj in earliest_objects])
            

            answer_source = []
            answer_source.append({'type': 'text', 'content': f'There are {len_video} frames in the {video_num} video in total.'})
            for obj in selected_candidates:
                obj_id, obj_name, time = obj
                answer_source.append({'type': 'text', 'content': f'{obj_name} last appears at frame {time}.'})
            obj_ids = [(obj[0], 0) for obj in selected_candidates]
            start_id = max(0, earliest_time - 5)
            end_id = min(len(trajectory) - 1, earliest_time + 5)
            clip = list(range(start_id, end_id))
            # clip_data = self.get_one_clip_data(clip, state_key, obj_ids)
            # answer_source.append({'type': 'text', 'content': f'Here is a video clip showing the last appearance times of objects in each option, spanning from frame {start_id} to frame {end_id}:'})
            # answer_source.append({'type': 'video', 'content': clip_data, 'state': state_key})

            related_frames = {state_key: clip}
            
            question_text = question_template.format(state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answers,
                'question_type': self.templates['object_last_appearance_earliest']['question_type'],
                'category': 'agent_explore',
                'subcategory': 'object_last_appearance_earliest',
                'capabilities': self.templates['object_last_appearance_earliest']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })
            
            for obj in selected_candidates:
                used_objects.add(obj[1])

            num += 1
            if num > qa_num:
                break
        

        return questions

    def _generate_last_latest_observation_time_questions(self, state_key: str, candidate_objs: List[Tuple[str, str, List[int]]], qa_num: int) -> List[Dict[str, Any]]:
        question_template = self.templates['object_last_appearance_latest']['question_template']
        questions = []

        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)

        used_objects = set()
        num = 0
        while True:
            available_candidates = [obj for obj in candidate_objs if obj[1] not in used_objects]
            
            if len(available_candidates) < 4:
                break
            
            num_choices = min(random.randint(4, 6), len(available_candidates))
            selected_candidates = random.sample(available_candidates, num_choices)
            
            # 按最后一次出现时间排序选中的候选物体
            selected_candidates.sort(key=lambda x: x[2])
            earliest_time = selected_candidates[0][2]  # 最早的时间
            latest_time = selected_candidates[-1][2]  # 最晚的时间
            latest_objects = [obj for obj in selected_candidates if obj[2] == latest_time]
            
            random.shuffle(selected_candidates)
            choices = [obj[1] for obj in selected_candidates]
            correct_answers = sorted([choices.index(obj[1]) for obj in latest_objects])
            
            answer_source = []
            answer_source.append({'type': 'text', 'content': f'There are {len_video} frames in the {video_num} video in total.'})
            for obj in selected_candidates:
                obj_id, obj_name, time = obj
                answer_source.append({'type': 'text', 'content': f'{obj_name} last appears at frame {time}.'})
            obj_ids = [(obj[0], 0) for obj in selected_candidates]
            start_id = max(0, earliest_time - 5)
            end_id = min(len(trajectory) - 1, earliest_time + 5)
            clip = list(range(start_id, end_id))
            # clip_data = self.get_one_clip_data(clip, state_key, obj_ids)
            # answer_source.append({'type': 'text', 'content': f'Here is a video clip showing the last appearance times of objects in each option, spanning from frame {start_id} to frame {end_id}:'})
            # answer_source.append({'type': 'video', 'content': clip_data, 'state': state_key})

            related_frames = {state_key: clip}
            
            question_text = question_template.format(state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answers,
                'question_type': self.templates['object_last_appearance_latest']['question_type'],
                'category': 'agent_explore',
                'subcategory': 'object_last_appearance_latest',
                'capabilities': self.templates['object_last_appearance_latest']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })
            
            for obj in selected_candidates:
                used_objects.add(obj[1])

            num += 1
            if num > qa_num:
                break

        return questions

    def _generate_object_appearance_time_questions(self, state_key: str) -> List[Dict[str, Any]]:
        available_keys = ['type', 'type_in_room', 'with_contents', 'in_receptacle', 'size', 'position', 'attribute']

        questions = []
        
        room_objects = self.room_objects[state_key]
        available_objects = {}
        for objects_in_room in room_objects.values():
            for obj in objects_in_room:
                obj_id = obj['objectId']
                if obj_id in self.object_names[state_key]:
                    obj_names = self.object_names[state_key][obj_id]
                    obj_name, obj_key = self._select_name(obj_names, available_keys) ###
                    if obj_name:
                        available_objects[obj_id] = obj_name
        
        trajectory = self.scene_data[state_key]['agent_trajectory']
        
        object_first_appearance = {}
        object_last_appearance = {}
        
        for step_idx, step in enumerate(trajectory):
            if 'visible_objects' in step:
                for obj_id in step['visible_objects']:
                    if obj_id in available_objects:
                        if obj_id not in object_first_appearance:
                            object_first_appearance[obj_id] = step_idx
                        object_last_appearance[obj_id] = step_idx
        
        if len(object_first_appearance) < 4:
            return questions
        
        candidate_objects_first = [(obj_id, available_objects[obj_id], time) 
                                  for obj_id, time in object_first_appearance.items()]
        candidate_objects_last = [(obj_id, available_objects[obj_id], time) 
                                 for obj_id, time in object_last_appearance.items()]
        
        candidate_objects_first = random.sample(candidate_objects_first, min(len(candidate_objects_first), 15))
        candidate_objects_last = random.sample(candidate_objects_last, min(len(candidate_objects_last), 15))
        
        candidate_objects_first.sort(key=lambda x: x[2])
        candidate_objects_last.sort(key=lambda x: x[2])
        
        # 生成第一次出现最早的物体问题
        questions.extend(self._generate_first_earliest_observation_time_questions(state_key, candidate_objects_first, 3))
        
        # 生成第一次出现最晚的物体问题
        questions.extend(self._generate_first_latest_observation_time_questions(state_key, candidate_objects_first, 3))
        
        # 生成最后一次出现最早的物体问题
        questions.extend(self._generate_last_earliest_observation_time_questions(state_key, candidate_objects_last, 3))
        
        # 生成最后一次出现最晚的物体问题
        questions.extend(self._generate_last_latest_observation_time_questions(state_key, candidate_objects_last, 3))
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    