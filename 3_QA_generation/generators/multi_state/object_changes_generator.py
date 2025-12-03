from enum import Flag
import os
import json
import random
import inspect
from typing import AnyStr, Dict, List, Any
from ..base_generator import BaseGenerator

class ObjectChangesGenerator(BaseGenerator):
    def __init__(self, scene_path: str):
        super().__init__(scene_path)
        template_path = os.path.join(os.path.dirname(__file__), '../../templates/multi_state/object_changes_template.json')
        with open(template_path, 'r', encoding='utf-8') as f:
            self.templates = json.load(f)['object_changes']
        
    def generate_questions(self) -> List[Dict[str, Any]]:
        questions = []
        
        states = self._get_ordered_states()

        for i in range(len(states) - 1):
            state1 = states[i]
            state2 = states[i+1]

            operations_log = self.scene_data[states[i+1]]['operations_log']
            
            questions.extend(self._generate_movement_distance_questions(operations_log, state1, state2))
            questions.extend(self._generate_receptacle_change_questions(operations_log, state1, state2))
            questions.extend(self._generate_object_visibility_change_questions(operations_log, state1, state2))
            questions.extend(self._generate_object_room_movement_questions(operations_log, state1, state2))
            questions.extend(self._generate_object_observation_time_questions(state1, state2))
            questions.extend(self._generate_object_move_in_a_room_questions(operations_log, state1, state2))
            questions.extend(self._generate_object_attribute_change_questions(operations_log, state1, state2))
            questions.extend(self._generate_specific_attribute_questions(operations_log, state1, state2))
        
        return questions
    

    def _generate_moved_object_questions(self, state1: str, state2: str, moved_objects: Dict, move_operations: Dict) -> List[Dict[str, Any]]:
        """Generate questions for objects that have moved."""
        question_template = self.templates['movement_distance']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)
        
        for obj_id, obj_name in moved_objects.items():
            op = move_operations[obj_id]
            pos1 = op['old_pos']
            pos2 = op['new_pos']

            distance = self.calculate_distance(pos1, pos2)
            unit = 'm'
            if distance < 0.1:
                distance *= 100
                unit = 'cm'
            
            num_choices = random.randint(4, 6)
            range_float = random.random()
            distance_range = [distance * range_float, distance * (1 + range_float)]
            choices, correct_choice_idx = self.create_range_choices(distance_range, distance, unit, num_choices - 1)
            if not choices:
                continue
            correct_value = choices[correct_choice_idx]
            choices.append(self.no_valid_option)
            random.shuffle(choices)
            correct_answer = choices.index(correct_value)

            # Generate related frames and answer source
            clips1 = self.get_object_visible_frames(state1, obj_id)
            clips2 = self.get_object_visible_frames(state2, obj_id)
            related_frames = {state1: clips1, state2: clips2}

            answer_source = []
            # answer_source = [{'type': 'text', 'content': f'The {video_num1} video has {video_len1} frames and the {video_num2} video has {video_len2} frames.'}]
            # clips = self.find_continous_clips(clips1)
            # clip = max(clips, key=len)
            # clip_data = self.get_one_clip_data(clip, state1, objs = [(obj_id, 0)])
            # start_frame = clip_data['frames'][0]['frame_id']
            # end_frame = clip_data['frames'][-1]['frame_id']
            # answer_source.append({'type': 'text', 'content': f'In the {video_num1} video, from frame {start_frame} to frame {end_frame} is a clip where {obj_name} can be seen:'})
            # answer_source.append({'type': 'video', 'video_data': clip_data})
            
            # clips = self.find_continous_clips(clips2)
            # clip = max(clips, key=len)
            # clip_data = self.get_one_clip_data(clip, state2, objs = [(obj_id, 0)])
            # start_frame = clip_data['frames'][0]['frame_id']
            # end_frame = clip_data['frames'][-1]['frame_id']
            # answer_source.append({'type': 'text', 'content': f'In the {video_num2} video, from frame {start_frame} to frame {end_frame} is a clip where {obj_name} can be seen:'})
            # answer_source.append({'type': 'video', 'video_data': clip_data})

            question = question_template.format(state1=state1, state2=state2, object_name=obj_name)
            questions.append({
                'question': question,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': 'single_choice',
                'category': 'object_changes',
                'subcategory': 'movement_distance',
                'capabilities': self.templates['movement_distance']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })
        
        return questions
    
    def _generate_unmoved_object_questions(self, state1: str, state2: str, unmoved_objects: Dict, num_moved_questions: int) -> List[Dict[str, Any]]:
        """Generate questions for objects that have not moved."""
        question_template = self.templates['movement_distance']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)
        
        # Select a subset of unmoved objects to generate questions for
        unmoved_object_ids = list(unmoved_objects.keys())
        selected_object_ids = random.sample(unmoved_object_ids, min(num_moved_questions, len(unmoved_object_ids)))
        
        for obj_id in selected_object_ids:
            obj_name = unmoved_objects[obj_id]
            num_choices = random.randint(4, 6)
            fake_answer = round(random.uniform(0.05, 10), 2)
            unit = 'm'
            if fake_answer < 0.1:
                fake_answer *= 100
                unit = 'cm'
            range_float = random.random()
            fake_range = [fake_answer * range_float, fake_answer * (1 + range_float)]
            choices, _ = self.create_range_choices(fake_range, fake_answer, unit, num_choices - 1)
            if not choices:
                continue
            choices.append(self.no_valid_option)
            random.shuffle(choices)
            correct_answer = choices.index(self.no_valid_option)

            # Generate related frames and answer source
            clips1 = self.get_object_visible_frames(state1, obj_id)
            clips2 = self.get_object_visible_frames(state2, obj_id)
            related_frames = {state1: clips1, state2: clips2}
            
            answer_source = []
            # answer_source = [{'type': 'text', 'content': f'The {video_num1} video has {video_len1} frames and the {video_num2} video has {video_len2} frames.'}]
            
            # # Add video clips for state1
            # clips = self.find_continous_clips(clips1)
            # if clips:
            #     clip = max(clips, key=len)
            #     clip_data = self.get_one_clip_data(clip, state1, objs = [(obj_id, 0)])
            #     start_frame = clip_data['frames'][0]['frame_id']
            #     end_frame = clip_data['frames'][-1]['frame_id']
            #     answer_source.append({'type': 'text', 'content': f'In the {video_num1} video, from frame {start_frame} to frame {end_frame} is a clip where {obj_name} can be seen:'})
            #     answer_source.append({'type': 'video', 'video_data': clip_data})
            
            # # Add video clips for state2
            # clips = self.find_continous_clips(clips2)
            # if clips:
            #     clip = max(clips, key=len)
            #     clip_data = self.get_one_clip_data(clip, state2, objs = [(obj_id, 0)])
            #     start_frame = clip_data['frames'][0]['frame_id']
            #     end_frame = clip_data['frames'][-1]['frame_id']
            #     answer_source.append({'type': 'text', 'content': f'In the {video_num2} video, from frame {start_frame} to frame {end_frame} is a clip where {obj_name} can be seen:'})
            #     answer_source.append({'type': 'video', 'video_data': clip_data})
            
            question_text = question_template.format(state1=state1, state2=state2, object_name=obj_name)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': 'single_choice',
                'category': 'object_changes',
                'subcategory': 'movement_distance',
                'capabilities': self.templates['movement_distance']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': True
            })
        
        return questions

    def _generate_movement_distance_questions(self, operations_log: Dict, state1: str, state2: str) -> List[Dict[str, Any]]:
        available_keys = ['type', 'type_in_room', 'in_receptacle', 'with_contents', 'size', 'attribute']

        questions = []
        
        all_objects_state1 = self.scene_data[state1]['objects_state']['objects_state']
        all_objects_state2 = self.scene_data[state2]['objects_state']['objects_state']
        
        objects_state1 = {}
        for obj_id, obj in all_objects_state1.items():
            if obj_id in self.object_names[state1]:
                obj_names = self.object_names[state1][obj_id]
                obj_name, obj_key = self._select_name(obj_names, available_keys, False)
                if obj_name:
                    objects_state1[obj_id] = obj_name

        objects_state2 = list(all_objects_state2.keys())

        common_objects = {}
        for obj_id1, obj_name1 in objects_state1.items():
            if obj_id1 in objects_state2:
                common_objects[obj_id1] = obj_name1

        operations = operations_log['operations']
        move_operations = {}
        moved_object_ids = []
        for op in operations:
            if op['action'] == 'PlaceObjectAtPoint':
                move_operations[op['object_id']] = op
                moved_object_ids.append(op['object_id'])

        moved_objects = {}
        unmoved_objects = {}
        for obj_id, obj_name in common_objects.items():
            if obj_id in moved_object_ids:
                moved_objects[obj_id] = obj_name
            else:
                unmoved_objects[obj_id] = (obj_name)

        questions.extend(self._generate_moved_object_questions(state1, 
                                                               state2,
                                                               moved_objects,
                                                               move_operations,))

        num_moved_questions = len(questions)
        questions.extend(self._generate_unmoved_object_questions(state1, 
                                                                 state2,
                                                                 unmoved_objects, 
                                                                 num_moved_questions))
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    

    def _generate_true_new_receptacle_questions(self, state1: str, state2: str, move_operations: Dict, all_object_names_state1: Dict, all_receptacle_names_state2: Dict, all_receptacles_names_state2: Dict, new: List) -> List[Dict[str, Any]]:
        question_template = self.templates['new_receptacle']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)

        for obj_id in new:
            if obj_id not in all_object_names_state1:
                continue
            obj_name = all_object_names_state1[obj_id]
            new_receptacle_id = move_operations[obj_id]['new_receptacle']
            new_receptacle_name = all_receptacles_names_state2[new_receptacle_id]

            num_choices = random.randint(4, 6)
            choices = [new_receptacle_name, self.no_valid_option]
            random.shuffle(all_receptacle_names_state2)
            for receptacle in all_receptacle_names_state2:
                if receptacle not in choices and len(choices) < num_choices:
                    choices.append(receptacle)
            random.shuffle(choices)
            correct_choice = choices.index(new_receptacle_name)

            
            related_frames = {state1: set(), state2: set()}
            clip1 = self.get_object_visible_frames(state1, obj_id)
            clip2 = self.get_object_visible_frames(state2, obj_id)
            related_frames[state1] = clip1
            related_frames[state2] = clip2
            
            answer_source = []

            question_text = question_template.format(state1=state1, state2=state2, object=obj_name,)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_choice,
                'question_type': self.templates['new_receptacle']['question_type'],
                'category': 'object_changes',
                'subcategory': 'position_change_detection',
                'capabilities': self.templates['new_receptacle']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })

        return questions
    
    def _generate_true_old_receptacle_questions(self, state1: str, state2: str, move_operations: Dict, all_object_names_state2: Dict, all_receptacle_names_state1: Dict, all_receptacles_names_state1: Dict, old: List) -> List[Dict[str, Any]]:
        question_template = self.templates['old_receptacle']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)

        for obj_id in old:
            if obj_id not in all_object_names_state2:
                continue
            obj_name = all_object_names_state2[obj_id]
            old_receptacle_id = move_operations[obj_id]['old_receptacle']
            old_receptacle_name = all_receptacles_names_state1[old_receptacle_id]

            num_choices = random.randint(4, 6)
            choices = [old_receptacle_name, self.no_valid_option]
            random.shuffle(all_receptacle_names_state1)
            for receptacle in all_receptacle_names_state1:
                if receptacle not in choices and len(choices) < num_choices:
                    choices.append(receptacle)
            random.shuffle(choices)
            correct_choice = choices.index(old_receptacle_name)

            
            related_frames = {state1: set(), state2: set()}
            clip1 = self.get_object_visible_frames(state1, obj_id)
            clip2 = self.get_object_visible_frames(state2, obj_id)
            related_frames[state1] = clip1
            related_frames[state2] = clip2

            answer_source = []

            question_text = question_template.format(state1=state1, state2=state2, object=obj_name,)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_choice,
                'question_type': self.templates['old_receptacle']['question_type'],
                'category': 'object_changes',
                'subcategory': 'position_change_detection',
                'capabilities': self.templates['old_receptacle']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })

        return questions
    
    def _generate_true_movement_between_receptacles_questions(self, state1: str, state2: str, move_operations: Dict, all_object_names_state1: Dict, all_receptacle_names_state1: Dict, all_receptacles_names_state1: Dict, all_receptacle_names_state2: Dict, all_receptacles_names_state2: Dict, both: List) -> List[Dict[str, Any]]:
        question_template = self.templates['movement_between_receptacles']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)

        for obj_id in both:
            if obj_id not in all_object_names_state1:
                continue
            obj_name = all_object_names_state1[obj_id]
            old_receptacle_id = move_operations[obj_id]['old_receptacle']
            new_receptacle_id = move_operations[obj_id]['new_receptacle']
            old_receptacle_name = all_receptacles_names_state1[old_receptacle_id]
            new_receptacle_name = all_receptacles_names_state2[new_receptacle_id]
            
            correct_answer_text = f"from {old_receptacle_name} to {new_receptacle_name}"
            
            num_choices = random.randint(4, 6)
            choices = [correct_answer_text, self.no_valid_option]
            
            all_receptacle_combinations = []
            for old_rec in all_receptacle_names_state1:
                for new_rec in all_receptacle_names_state2:
                    combination = f"from {old_rec} to {new_rec}"
                    if combination != correct_answer_text:
                        all_receptacle_combinations.append(combination)
            
            random.shuffle(all_receptacle_combinations)
            for combination in all_receptacle_combinations:
                if combination not in choices and len(choices) < num_choices:
                    choices.append(combination)
            
            random.shuffle(choices)
            correct_choice = choices.index(correct_answer_text)
            
            
            related_frames = {state1: set(), state2: set()}
            clip1 = self.get_object_visible_frames(state1, obj_id)
            clip2 = self.get_object_visible_frames(state2, obj_id)
            related_frames[state1] = clip1
            related_frames[state2] = clip2

            answer_source = []

            question_text = question_template.format(state1=state1, state2=state2, object=obj_name)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_choice,
                'question_type': self.templates['movement_between_receptacles']['question_type'],
                'category': 'object_changes',
                'subcategory': 'position_change_detection',
                'capabilities': self.templates['movement_between_receptacles']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })

        return questions
    
    def _generate_true_receptacle_change_questions(self, state1: str, state2: str, move_operations: Dict, all_object_names_state1: Dict, all_object_names_state2: Dict, all_receptacles_names_state1: Dict, all_receptacles_names_state2: Dict) -> List[Dict[str, Any]]:
        both = []
        old = []
        new = []
        for op in move_operations.values():
            obj_id = op['object_id']
            old_receptacle_id = op['old_receptacle']
            new_receptacle_id = op['new_receptacle']
            old_receptacle_name = all_receptacles_names_state1.get(old_receptacle_id)
            new_receptacle_name = all_receptacles_names_state2.get(new_receptacle_id)
            if old_receptacle_name and new_receptacle_name:
                both.append(obj_id)
            elif old_receptacle_name:
                old.append(obj_id)
            elif new_receptacle_name:
                new.append(obj_id)
        
        all_receptacle_names_state1 = list(all_receptacles_names_state1.values())
        all_receptacle_names_state2 = list(all_receptacles_names_state2.values())

        new_receptacle_qa = self._generate_true_new_receptacle_questions(state1, 
                                                                    state2, 
                                                                    move_operations, 
                                                                    all_object_names_state1, 
                                                                    all_receptacle_names_state2, 
                                                                    all_receptacles_names_state2, 
                                                                    new)

        old_receptacle_qa = self._generate_true_old_receptacle_questions(state1, 
                                                                    state2, 
                                                                    move_operations, 
                                                                    all_object_names_state2, 
                                                                    all_receptacle_names_state1, 
                                                                    all_receptacles_names_state1, 
                                                                    old)

        movement_qa = self._generate_true_movement_between_receptacles_questions(state1, 
                                                                            state2, 
                                                                            move_operations, 
                                                                            all_object_names_state1, 
                                                                            all_receptacle_names_state1, 
                                                                            all_receptacles_names_state1, 
                                                                            all_receptacle_names_state2, 
                                                                            all_receptacles_names_state2, 
                                                                            both)
        
        return new_receptacle_qa, old_receptacle_qa, movement_qa

    def _generate_false_new_receptacle_questions(self, state1: str, state2: str, all_object_names_state1: Dict, all_receptacles_names_state2: Dict, unmoved_pickupable_obj_ids: List, qa_num: int) -> List[Dict[str, Any]]:
        question_template = self.templates['new_receptacle']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)

        random.shuffle(unmoved_pickupable_obj_ids)
        used_obj_ids = set()

        num = 0
        max_num = max(1, qa_num)
        for obj_id in unmoved_pickupable_obj_ids:
            if num >= max_num or obj_id in used_obj_ids:
                continue
            
            obj_name = all_object_names_state1[obj_id]
            used_obj_ids.add(obj_id)
            
            num_choices = random.randint(4, 6)
            choices = [self.no_valid_option]
            random.shuffle(list(all_receptacles_names_state2.values()))
            for receptacle in list(all_receptacles_names_state2.values()):
                if receptacle not in choices and len(choices) < num_choices:
                    choices.append(receptacle)
            random.shuffle(choices)
            correct_choice = choices.index(self.no_valid_option)
            
            
            related_frames = {state1: set(), state2: set()}
            clip1 = self.get_object_visible_frames(state1, obj_id)
            clip2 = self.get_object_visible_frames(state2, obj_id)
            related_frames[state1] = clip1
            related_frames[state2] = clip2

            answer_source = []

            question_text = question_template.format(state1=state1, state2=state2, object=obj_name)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_choice,
                'question_type': self.templates['new_receptacle']['question_type'],
                'category': 'object_changes',
                'subcategory': 'position_change_detection',
                'capabilities': self.templates['new_receptacle']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': True
            })
            num += 1

        return questions
    
    def _generate_false_old_receptacle_questions(self, state1: str, state2: str, all_object_names_state1: Dict, all_receptacles_names_state1: Dict, unmoved_pickupable_obj_ids: List, qa_num: int) -> List[Dict[str, Any]]:
        question_template = self.templates['old_receptacle']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)

        random.shuffle(unmoved_pickupable_obj_ids)
        used_obj_ids = set()

        num = 0
        max_num = max(1, qa_num)
        for obj_id in unmoved_pickupable_obj_ids:
            if num >= max_num or obj_id in used_obj_ids:
                continue
            
            obj_name = all_object_names_state1[obj_id]
            used_obj_ids.add(obj_id)
            
            num_choices = random.randint(4, 6)
            choices = [self.no_valid_option]
            random.shuffle(list(all_receptacles_names_state1.values()))
            for receptacle in list(all_receptacles_names_state1.values()):
                if receptacle not in choices and len(choices) < num_choices:
                    choices.append(receptacle)
            random.shuffle(choices)
            correct_choice = choices.index(self.no_valid_option)
            
            
            related_frames = {state1: set(), state2: set()}
            clip1 = self.get_object_visible_frames(state1, obj_id)
            clip2 = self.get_object_visible_frames(state2, obj_id)
            related_frames[state1] = clip1
            related_frames[state2] = clip2

            answer_source = []

            question_text = question_template.format(state1=state1, state2=state2, object=obj_name)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_choice,
                'question_type': self.templates['old_receptacle']['question_type'],
                'category': 'object_changes',
                'subcategory': 'position_change_detection',
                'capabilities': self.templates['old_receptacle']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': True
            })
            num += 1

        return questions
    
    def _generate_false_movement_between_receptacles_questions(self, state1: str, state2: str, all_object_names_state1: Dict, all_receptacles_names_state1: Dict, all_receptacles_names_state2: Dict, unmoved_pickupable_obj_ids: List, qa_num: int) -> List[Dict[str, Any]]:
        question_template = self.templates['movement_between_receptacles']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)

        random.shuffle(unmoved_pickupable_obj_ids)
        used_obj_ids = set()

        num = 0
        max_num = max(1, qa_num)
        for obj_id in unmoved_pickupable_obj_ids:
            if num >= max_num or obj_id in used_obj_ids:
                continue
            
            obj_name = all_object_names_state1[obj_id]
            used_obj_ids.add(obj_id)
            
            num_choices = random.randint(4, 6)
            choices = [self.no_valid_option]
            
            all_receptacle_combinations = []
            receptacle_names_1 = list(all_receptacles_names_state1.values())
            receptacle_names_2 = list(all_receptacles_names_state2.values())
            for old_rec in receptacle_names_1[:3]:  # 限制组合数量
                for new_rec in receptacle_names_2[:3]:
                    combination = f"from {old_rec} to {new_rec}"
                    all_receptacle_combinations.append(combination)
            
            random.shuffle(all_receptacle_combinations)
            for combination in all_receptacle_combinations:
                if combination not in choices and len(choices) < num_choices:
                    choices.append(combination)
            
            random.shuffle(choices)
            correct_choice = choices.index(self.no_valid_option)
            
            
            related_frames = {state1: set(), state2: set()}
            clip1 = self.get_object_visible_frames(state1, obj_id)
            clip2 = self.get_object_visible_frames(state2, obj_id)
            related_frames[state1] = clip1
            related_frames[state2] = clip2

            answer_source = []

            question_text = question_template.format(state1=state1, state2=state2, object=obj_name)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_choice,
                'question_type': self.templates['movement_between_receptacles']['question_type'],
                'category': 'object_changes',
                'subcategory': 'position_change_detection',
                'capabilities': self.templates['movement_between_receptacles']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': True
            })
            num += 1

        return questions

    def _generate_false_receptacle_change_questions(self, state1: str, state2: str, move_operations: Dict, all_object_names_state1: Dict, all_object_names_state2: Dict, common_obj_ids: List, all_receptacles_names_state1: Dict, all_receptacles_names_state2: Dict, num_new_receptacle_qa: int, num_old_receptacle_qa: int, num_movement_qa: int) -> List[Dict[str, Any]]:
        all_objects_state1 = self.scene_data[state1]['objects_state']['objects_state']
        
        unmoved_obj_ids = list(set(common_obj_ids) - set(list(move_operations.keys())))
        unmoved_pickupable_obj_ids = [obj_id for obj_id in unmoved_obj_ids if obj_id in all_object_names_state1 and all_objects_state1[obj_id]['pickupable']]
        
        f_new_receptacle_qa = self._generate_false_new_receptacle_questions(state1,
                                                                       state2,
                                                                       all_object_names_state1,
                                                                       all_receptacles_names_state2,
                                                                       unmoved_pickupable_obj_ids,
                                                                       num_new_receptacle_qa)
        
        f_old_receptacle_qa = self._generate_false_old_receptacle_questions(state1,
                                                                       state2,
                                                                       all_object_names_state1,
                                                                       all_receptacles_names_state1,
                                                                       unmoved_pickupable_obj_ids,
                                                                       num_old_receptacle_qa)
        
        f_movement_qa = self._generate_false_movement_between_receptacles_questions(state1,
                                                                                     state2,
                                                                                     all_object_names_state1,
                                                                                     all_receptacles_names_state1,
                                                                                     all_receptacles_names_state2,
                                                                                     unmoved_pickupable_obj_ids,
                                                                                     num_movement_qa)
        
        return f_new_receptacle_qa, f_old_receptacle_qa, f_movement_qa

    def _generate_receptacle_change_questions(self, operations_log: Dict, state1: str, state2: str) -> List[Dict[str, Any]]:
        obj_available_keys = ['type', 'type_in_room']
        receptacle_available_keys = ['type', 'type_in_room', 'in_receptacle', 'size', 'attribute']

        questions = []
        
        all_objects_state1 = self.scene_data[state1]['objects_state']['objects_state']
        all_objects_state2 = self.scene_data[state2]['objects_state']['objects_state']
        common_obj_ids = list(set(list(all_objects_state1.keys())) & set(list(all_objects_state2.keys())))

        all_object_names_state1 = {}
        all_receptacles_names_state1 = {}
        for obj_id, obj in all_objects_state1.items():
            if obj['receptacle']:
                obj_names = self.object_names[state1][obj_id]
                obj_name, obj_key = self._select_name(obj_names, receptacle_available_keys)
                if obj_name:
                    all_receptacles_names_state1[obj_id] = obj_name
            obj_names = self.object_names[state1][obj_id]
            obj_name, obj_key = self._select_name(obj_names, obj_available_keys, False)
            if obj_name:
                all_object_names_state1[obj_id] = obj_name
            
        all_object_names_state2 = {}
        all_receptacles_names_state2 = {}
        for obj_id, obj in all_objects_state2.items():
            if obj['receptacle']:
                obj_names = self.object_names[state2][obj_id]
                obj_name, obj_key = self._select_name(obj_names, receptacle_available_keys)
                if obj_name:
                    all_receptacles_names_state2[obj_id] = obj_name
            obj_names = self.object_names[state2][obj_id]
            obj_name, obj_key = self._select_name(obj_names, obj_available_keys, False)
            if obj_name:
                all_object_names_state2[obj_id] = obj_name

        operations = operations_log['operations']
        move_operations = {}
        for op in operations:
            if op['action'] == 'PlaceObjectAtPoint' and op['object_id'] in common_obj_ids:
                move_operations[op['object_id']] = op

        new_receptacle_qa, old_receptacle_qa, movement_qa = self._generate_true_receptacle_change_questions(state1, 
                                                                    state2, 
                                                                    move_operations, 
                                                                    all_object_names_state1, 
                                                                    all_object_names_state2, 
                                                                    all_receptacles_names_state1, 
                                                                    all_receptacles_names_state2)

        f_new_receptacle_qa, f_old_receptacle_qa, f_movement_qa = self._generate_false_receptacle_change_questions(state1,
                                                          state2,
                                                          move_operations,
                                                            all_object_names_state1,
                                                            all_object_names_state2, 
                                                            common_obj_ids,
                                                            all_receptacles_names_state1,
                                                            all_receptacles_names_state2,
                                                            len(new_receptacle_qa),
                                                            len(old_receptacle_qa),
                                                            len(movement_qa))
        
        questions = new_receptacle_qa + old_receptacle_qa + movement_qa + f_new_receptacle_qa + f_old_receptacle_qa + f_movement_qa

        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    
    
    def _generate_new_visible_objects_questions(self, state1: str, state2: str, all_available_objects: Dict, new_available_objects: Dict, lost_available_objects: Dict) -> List[Dict[str, Any]]:
        question_template = self.templates['new_visible_objects']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)

        candidate_options = list(lost_available_objects.items()) + list(all_available_objects.items())
        left_new_available_objects = list(new_available_objects.keys())

        while left_new_available_objects:
            num_choices = random.randint(4, 6)
            num_correct_choices = random.randint(1, min(num_choices, len(left_new_available_objects)))
            random.shuffle(left_new_available_objects)
            correct_obj_ids = left_new_available_objects[:num_correct_choices]
            correct_answers = [new_available_objects[obj_id] for obj_id in correct_obj_ids]
            choices = correct_answers.copy()
            left_new_available_objects = left_new_available_objects[num_correct_choices:]

            random.shuffle(candidate_options)
            choice_obj_ids = {}
            for obj_id, obj_name in candidate_options[:num_choices - num_correct_choices]:
                choices.append(obj_name)
                choice_obj_ids[obj_name] = obj_id
            
            for obj_id in correct_obj_ids:
                choice_obj_ids[new_available_objects[obj_id]] = obj_id
            
            is_hallucination = False
            if num_correct_choices <= 2 and random.random() < 1/3:
                is_hallucination = True
                for answer in correct_answers:
                    choices.remove(answer)
                choices.append(self.no_valid_option)
                correct_answers = [self.no_valid_option]
            else:
                incorrect_choices = [choice for choice in choices if choice not in correct_answers]
                if incorrect_choices:
                    choices.remove(random.choice(incorrect_choices))
                    choices.append(self.no_valid_option)
            
            random.shuffle(choices)
            correct_choices = [i for i, choice in enumerate(choices) if choice in correct_answers]
            
            related_frames = {state1: set(), state2: set()}
            for obj_name in choices:
                if obj_name == self.no_valid_option:
                    continue
                obj_id = choice_obj_ids[obj_name]
                frames1 = self.get_object_visible_frames(state1, obj_id)
                frames2 = self.get_object_visible_frames(state2, obj_id)
                related_frames[state1].update(frames1)
                related_frames[state2].update(frames2)
            related_frames[state1] = sorted(list(related_frames[state1]))
            related_frames[state2] = sorted(list(related_frames[state2]))
            
            answer_source = []

            question_text = question_template.format(state1=state1,state2=state2,)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_choices,
                'question_type': 'multiple_choice',
                'category': 'object_changes',
                'subcategory': 'new_visible_objects',
                'capabilities': self.templates['new_visible_objects']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })

        return questions

    def _generate_lost_visible_objects_questions(self, state1: str, state2: str, all_available_objects: Dict, new_available_objects: Dict, lost_available_objects: Dict) -> List[Dict[str, Any]]:
        question_template = self.templates['lost_visible_objects']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)

        candidate_options = list(new_available_objects.items()) + list(all_available_objects.items())
        left_lost_available_objects = list(lost_available_objects.keys())

        while left_lost_available_objects:
            num_choices = random.randint(4, 6)
            num_correct_choices = random.randint(1, min(num_choices, len(left_lost_available_objects)))
            random.shuffle(left_lost_available_objects)
            correct_obj_ids = left_lost_available_objects[:num_correct_choices]
            correct_answers = [lost_available_objects[obj_id] for obj_id in correct_obj_ids]
            choices = correct_answers.copy()
            left_lost_available_objects = left_lost_available_objects[num_correct_choices:]

            random.shuffle(candidate_options)
            choice_obj_ids = {}
            for obj_id, obj_name in candidate_options[:num_choices - num_correct_choices]:
                choices.append(obj_name)
                choice_obj_ids[obj_name] = obj_id
            
            for obj_id in correct_obj_ids:
                choice_obj_ids[lost_available_objects[obj_id]] = obj_id
            
            is_hallucination = False
            if num_correct_choices <= 2 and random.random() < 1/3:
                is_hallucination = True
                for answer in correct_answers:
                    choices.remove(answer)
                choices.append(self.no_valid_option)
                correct_answers = [self.no_valid_option]
            else:
                incorrect_choices = [choice for choice in choices if choice not in correct_answers]
                if incorrect_choices:
                    choices.remove(random.choice(incorrect_choices))
                    choices.append(self.no_valid_option)
            
            random.shuffle(choices)
            correct_choices = [i for i, choice in enumerate(choices) if choice in correct_answers]
            
            related_frames = {state1: set(), state2: set()}
            for obj_name in choices:
                if obj_name == self.no_valid_option:
                    continue
                obj_id = choice_obj_ids[obj_name]
                frames1 = self.get_object_visible_frames(state1, obj_id)
                frames2 = self.get_object_visible_frames(state2, obj_id)
                related_frames[state1].update(frames1)
                related_frames[state2].update(frames2)
            related_frames[state1] = sorted(list(related_frames[state1]))
            related_frames[state2] = sorted(list(related_frames[state2]))
            
            answer_source = []
            
            question_text = question_template.format(state1=state1,state2=state2,)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_choices,
                'question_type': 'multiple_choice',
                'category': 'object_changes',
                'subcategory': 'lost_visible_objects',
                'capabilities': self.templates['lost_visible_objects']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })

        return questions

    def _generate_object_visibility_change_questions(self, operations_log: Dict, state1: str, state2: str) -> List[Dict[str, Any]]:
        obj_available_keys = ['type', 'type_in_room', 'in_receptacle', 'with_contents', 'size', 'attribute']

        questions = []

        slice_objects = []
        broken_objects = []
        operations = operations_log['operations']
        for op in operations:
            if op['action'] == "BreakObject":
                broken_objects.append(op['object_id'])
            elif op['action'] == "SliceObject":
                slice_objects.append(op['object_id'])

        all_objects_state1_ = self.scene_data[state1]['objects_state']['objects_state']
        all_objects_state1 = {}
        for obj_id, obj in all_objects_state1_.items():
            if obj_id in broken_objects or obj_id in slice_objects:
                continue
            else:
                all_objects_state1[obj_id] = obj
        all_objects_state2_ = self.scene_data[state2]['objects_state']['objects_state']
        all_objects_state2 = {}
        for obj_id, obj in all_objects_state2_.items():
            if obj_id in broken_objects:
                continue
            else:
                flag = False
                for slice_obj in slice_objects:
                    if slice_obj in obj_id:
                        flag = True
                        break
                if flag:
                    continue
                else:
                    all_objects_state2[obj_id] = obj

        common_objects = set(all_objects_state1.keys()) & set(all_objects_state2.keys())

        new_objects = set(all_objects_state2.keys()) - set(all_objects_state1.keys())
        lost_objects = set(all_objects_state1.keys()) - set(all_objects_state2.keys())

        new_available_objects = {}
        for obj_id in new_objects:
            if 'Sliced' in obj_id:
                continue
            obj_names = self.object_names[state2][obj_id]
            obj_name, obj_key = self._select_name(obj_names, obj_available_keys)
            if obj_name:
                new_available_objects[obj_id] = obj_name

        lost_available_objects = {}
        for obj_id in lost_objects:
            obj_names = self.object_names[state1][obj_id]
            obj_name, obj_key = self._select_name(obj_names, obj_available_keys)
            if obj_name:
                lost_available_objects[obj_id] = obj_name

        all_available_objects = {}
        for obj_id in common_objects:
            obj_names = self.object_names[state1][obj_id]
            obj_name1, obj_key1 = self._select_name(obj_names, obj_available_keys)
            obj_names = self.object_names[state2][obj_id]
            obj_name2, obj_key2 = self._select_name(obj_names, obj_available_keys)
            if obj_name1 and obj_name2 and obj_name1 == obj_name2:
                all_available_objects[obj_id] = obj_name1
    
        questions.extend(self._generate_new_visible_objects_questions(state1, state2, all_available_objects, new_available_objects, lost_available_objects))
        questions.extend(self._generate_lost_visible_objects_questions(state1, state2, all_available_objects, new_available_objects, lost_available_objects))

        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions


    def _generate_object_room_movement_questions(self, operations_log: Dict, state1: str, state2: str) -> List[Dict[str, Any]]:
        obj_available_keys = ['type', 'type_in_room', 'in_receptacle', 'with_contents', 'size', 'attribute']

        question_template = self.templates['objects_room_movement']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)

        all_objects_state1 = self.scene_data[state1]['objects_state']['objects_state']
        all_objects_state2 = self.scene_data[state2]['objects_state']['objects_state']
        common_obj_ids = list(set(list(all_objects_state1.keys())) & set(list(all_objects_state2.keys())))

        all_object_names_state1 = {}
        all_object_names_state2 = {}
        for obj_id in common_obj_ids:
            if obj_id in self.object_names[state1]:
                obj_names = self.object_names[state1][obj_id]
                obj_name, obj_key = self._select_name(obj_names, obj_available_keys, False)
                if obj_name:
                    all_object_names_state1[obj_id] = obj_name
                    
            if obj_id in self.object_names[state2]:
                obj_names = self.object_names[state2][obj_id]
                obj_name, obj_key = self._select_name(obj_names, obj_available_keys, False)
                if obj_name:
                    all_object_names_state2[obj_id] = obj_name

        operations = operations_log['operations']
        move_operations = {}
        for op in operations:
            if op['action'] == 'PlaceObjectAtPoint' and op['object_id'] in common_obj_ids:
                move_operations[op['object_id']] = op

        room_changed_objects = []
        room_unchanged_objects = []
        for obj_id in common_obj_ids:
            if obj_id not in all_object_names_state1 or obj_id not in all_object_names_state2:
                continue
                
            obj_state1 = all_objects_state1[obj_id]
            obj_state2 = all_objects_state2[obj_id]
            
            room1 = obj_state1.get('room_name')
            room2 = obj_state2.get('room_name')
                        
            if obj_id in move_operations and room1 != room2:
                room_changed_objects.append(obj_id)
            else:
                room_unchanged_objects.append(obj_id)
        
        if len(room_changed_objects) + len(room_unchanged_objects) >= 3:
            
            all_available_objects = room_changed_objects + room_unchanged_objects
            
            num = 0
            while True:
                if len(all_available_objects) < 3:
                    break
                
                num_choices = min(random.randint(3, 5), len(all_available_objects))
                selected_objects = random.sample(all_available_objects, num_choices)
                for obj in selected_objects:
                    all_available_objects.remove(obj)
                choices = [all_object_names_state1[obj_id] for obj_id in selected_objects] + [self.no_valid_option]
                random.shuffle(choices)
                correct_indices = []
                for obj_id in selected_objects:
                    if obj_id in room_changed_objects:
                        correct_indices.append(choices.index(all_object_names_state1[obj_id]))
                if not correct_indices:
                    correct_indices = [choices.index(self.no_valid_option)]
                    is_hallucination = True
                else:
                    is_hallucination = False
                
                
                related_frames = {state1: set(), state2: set()}
                for obj_id in selected_objects:
                    frames1 = self.get_object_visible_frames(state1, obj_id)
                    frames2 = self.get_object_visible_frames(state2, obj_id)
                    related_frames[state1].update(frames1)
                    related_frames[state2].update(frames2)
                related_frames[state1] = sorted(list(related_frames[state1]))
                related_frames[state2] = sorted(list(related_frames[state2]))
                
                answer_source = []

                question_text = question_template.format(state1=state1, state2=state2)
                questions.append({
                    'question': question_text,
                    'choices': choices,
                    'correct_answer': correct_indices,
                    'question_type': self.templates['objects_room_movement']['question_type'],
                    'category': 'object_changes',
                    'subcategory': 'room_movement',
                    'capabilities': self.templates['objects_room_movement']['capabilities'],
                    'answer_source': answer_source,
                    'related_frames': related_frames,
                    'hallucination': is_hallucination
                })

                num += 1
                if num >= 10:
                    break

        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions


    def _generate_longest_observed_object_type_questions(self, state1: str, state2: str, object_observation_times: Dict, available_objects: Dict, qa_num: int) -> List[Dict[str, Any]]:
        question_template = self.templates['longest_observed_object_type']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)

        candidate_objects = [(obj_id, available_objects[obj_id], times) for obj_id, times in object_observation_times.items()]
        used_objects = set()
        num = 0
        while True:
            available_candidates = [obj for obj in candidate_objects if obj[1] not in used_objects]
            if len(available_candidates) < 4:
                break
            
            num_choices = min(random.randint(4, 6), len(available_candidates))
            selected_candidates = random.sample(available_candidates, num_choices)
            
            selected_candidates.sort(key=lambda x: x[2]['total'])
            longest_observed_obj = selected_candidates[-1]
            
            choices = [obj[1] for obj in selected_candidates]
            random.shuffle(choices)
            correct_answer = choices.index(longest_observed_obj[1])
            
            related_frames = {state1: set(), state2: set()}
            for obj_id, _, _ in selected_candidates:
                related_frames[state1].update(object_observation_times[obj_id]['state1'])
                related_frames[state2].update(object_observation_times[obj_id]['state2'])
            related_frames[state1] = sorted(list(related_frames[state1]))
            related_frames[state2] = sorted(list(related_frames[state2]))
            
            answer_source = []
            
            question_text = question_template.format(state1=state1, state2=state2, obj=longest_observed_obj[1])
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['longest_observed_object_type']['question_type'],
                'category': 'object_changes',
                'subcategory': 'longest_observed_object_type',
                'capabilities': self.templates['longest_observed_object_type']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })

            for obj in selected_candidates:
                used_objects.add(obj[1])

            num += 1
            if num >= qa_num:
                break

        return questions
    
    def _generate_shortest_observed_object_type_questions(self, state1: str, state2: str, object_observation_times: Dict, available_objects: Dict, qa_num: int) -> List[Dict[str, Any]]:
        question_template = self.templates['shortest_observed_object_type']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)

        candidate_objects = [(obj_id, available_objects[obj_id], times) for obj_id, times in object_observation_times.items()]
        used_objects = set()
        num = 0
        while True:
            available_candidates = [obj for obj in candidate_objects if obj[1] not in used_objects]
            if len(available_candidates) < 4:
                break
            
            num_choices = min(random.randint(4, 6), len(available_candidates))
            selected_candidates = random.sample(available_candidates, num_choices)
            
            # 按总观察时间排序
            selected_candidates.sort(key=lambda x: x[2]['total'])
            shortest_observed_obj = selected_candidates[0]
            
            choices = [obj[1] for obj in selected_candidates]
            random.shuffle(choices)
            correct_answer = choices.index(shortest_observed_obj[1])
            
            related_frames = {state1: set(), state2: set()}
            for obj_id, _, _ in selected_candidates:  # 遍历所有选项物体
                related_frames[state1].update(object_observation_times[obj_id]['state1'])
                related_frames[state2].update(object_observation_times[obj_id]['state2'])
            related_frames[state1] = sorted(list(related_frames[state1]))  # 转换为有序列表
            related_frames[state2] = sorted(list(related_frames[state2]))  # 转换为有序列表
            
            answer_source = []
            
            question_text = question_template.format(state1=state1, state2=state2, obj=shortest_observed_obj[1])
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['shortest_observed_object_type']['question_type'],
                'category': 'object_changes',
                'subcategory': 'shortest_observed_object_type',
                'capabilities': self.templates['shortest_observed_object_type']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })
            
            for obj in selected_candidates:
                used_objects.add(obj[1])

            num += 1
            if num >= qa_num:
                break
        
        return questions
    
    def _generate_largest_diff_observation_time_questions(self, state1: str, state2: str, object_observation_times: Dict, available_objects: Dict, qa_num: int) -> List[Dict[str, Any]]:
        question_template = self.templates['largest_diff_observation_time']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)

        candidate_objects = [(obj_id, available_objects[obj_id], times) for obj_id, times in object_observation_times.items()]
        used_objects = set()
        num = 0
        while True:
            available_candidates = [obj for obj in candidate_objects if obj[1] not in used_objects]
            if len(available_candidates) < 4:
                break
            
            num_choices = min(random.randint(4, 6), len(available_candidates))
            selected_candidates = random.sample(available_candidates, num_choices)
            
            selected_candidates.sort(key=lambda x: x[2]['diff'])
            largest_diff_obj = selected_candidates[-1]
            
            choices = [obj[1] for obj in selected_candidates]
            random.shuffle(choices)
            correct_answer = choices.index(largest_diff_obj[1])
            
            related_frames = {state1: set(), state2: set()}
            for obj_id, _, _ in selected_candidates:  # 遍历所有选项物体
                related_frames[state1].update(object_observation_times[obj_id]['state1'])
                related_frames[state2].update(object_observation_times[obj_id]['state2'])
            related_frames[state1] = sorted(list(related_frames[state1]))  # 转换为有序列表
            related_frames[state2] = sorted(list(related_frames[state2]))  # 转换为有序列表
            
            answer_source = []
            
            question_text = question_template.format(state1=state1, state2=state2)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['largest_diff_observation_time']['question_type'],
                'category': 'object_changes',
                'subcategory': 'largest_diff_observation_time',
                'capabilities': self.templates['largest_diff_observation_time']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })
            
            for obj in selected_candidates:
                used_objects.add(obj[1])

            num += 1
            if num >= qa_num:
                break

        return questions
    
    def _generate_smallest_diff_observation_time_questions(self, state1: str, state2: str, object_observation_times: Dict, available_objects: Dict, qa_num: int) -> List[Dict[str, Any]]:
        question_template = self.templates['smallest_diff_observation_time']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)

        candidate_objects = [(obj_id, available_objects[obj_id], times) for obj_id, times in object_observation_times.items()]
        used_objects = set()
        num = 0
        while True:
            available_candidates = [obj for obj in candidate_objects if obj[1] not in used_objects]
            if len(available_candidates) < 4:
                break
            
            num_choices = min(random.randint(4, 6), len(available_candidates))
            selected_candidates = random.sample(available_candidates, num_choices)
            
            selected_candidates.sort(key=lambda x: x[2]['diff'])
            smallest_diff_obj = selected_candidates[0]
            
            choices = [obj[1] for obj in selected_candidates]
            random.shuffle(choices)
            correct_answer = choices.index(smallest_diff_obj[1])
            
            related_frames = {state1: set(), state2: set()}
            for obj_id, _, _ in selected_candidates:  # 遍历所有选项物体
                related_frames[state1].update(object_observation_times[obj_id]['state1'])
                related_frames[state2].update(object_observation_times[obj_id]['state2'])
            related_frames[state1] = sorted(list(related_frames[state1]))  # 转换为有序列表
            related_frames[state2] = sorted(list(related_frames[state2]))  # 转换为有序列表
            
            answer_source = []
            
            question_text = question_template.format(state1=state1, state2=state2)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['smallest_diff_observation_time']['question_type'],
                'category': 'object_changes',
                'subcategory': 'smallest_diff_observation_time',
                'capabilities': self.templates['smallest_diff_observation_time']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })
            
            for obj in selected_candidates:
                used_objects.add(obj[1])

            num += 1
            if num >= qa_num:
                break

        return questions

    def _generate_object_observation_time_questions(self, state1: str, state2: str) -> List[Dict[str, Any]]:
        available_keys = ['type', 'type_in_room', 'with_contents', 'in_receptacle', 'size', 'attribute']

        questions = []
        
        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        
        common_objects = list(self.object_names[state1].keys() & self.object_names[state2].keys())
        
        available_objects = {}
        for obj_id in common_objects:
            obj_names = self.object_names[state1][obj_id]
            obj_name1, obj_key1 = self._select_name(obj_names, available_keys)
            obj_names = self.object_names[state2][obj_id]
            obj_name2, obj_key2 = self._select_name(obj_names, available_keys)
            if obj_name1 and obj_name2 and obj_name1 == obj_name2:
                available_objects[obj_id] = obj_name1
        
        object_observation_times = {}
        
        for i, step in enumerate(trajectory1):
            visible_objects = step.get('visible_objects', [])
            for obj_id in visible_objects:
                if obj_id in available_objects.keys():
                    if obj_id not in object_observation_times:
                        object_observation_times[obj_id] = {'state1': [], 'state2': [], 'total': 0, 'diff': 0}
                    object_observation_times[obj_id]['state1'].append(i)
        
        for i, step in enumerate(trajectory2):
            visible_objects = step.get('visible_objects', [])
            for obj_id in visible_objects:
                if obj_id in available_objects.keys():
                    if obj_id not in object_observation_times:
                        object_observation_times[obj_id] = {'state1': [], 'state2': [], 'total': 0, 'diff': 0}
                    object_observation_times[obj_id]['state2'].append(i)
        
        for obj_id in object_observation_times:
            times = object_observation_times[obj_id]
            times['total'] = len(times['state1']) + len(times['state2'])
            times['diff'] = abs(len(times['state1']) - len(times['state2']))
        
        questions.extend(self._generate_longest_observed_object_type_questions(state1, state2, object_observation_times, available_objects, 5))
        questions.extend(self._generate_shortest_observed_object_type_questions(state1, state2, object_observation_times, available_objects, 5))
        questions.extend(self._generate_largest_diff_observation_time_questions(state1, state2, object_observation_times, available_objects, 5))
        questions.extend(self._generate_smallest_diff_observation_time_questions(state1, state2, object_observation_times, available_objects, 5))

        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    

    def _generate_object_move_in_a_room_questions(self, operations_log: Dict, state1: str, state2: str):
        obj_available_keys = ['type', 'with_contents', 'in_receptacle', 'size', 'attribute']
        room_available_keys = ['type', 'biggest_room', 'smallest_room', 'biggest_type', 'smallest_type', 'shape', 'unique_object']
        
        question_template = self.templates['room_position_changes']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        len_video1 = len(trajectory1)
        len_video2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)

        rooms = self.scene_data['room_static']['room_static_details']
        available_rooms = {}
        for room in rooms:
            room_id = room['room_name']
            
            if room_id in self.room_names[state1]:
                room_names = self.room_names[state1][room_id]
                room_name_state1, room_key1 = self._select_name(room_names, room_available_keys)
                room_names = self.room_names[state2][room_id]
                room_name_state2, room_key2 = self._select_name(room_names, room_available_keys)
                if room_name_state1 is not None and room_name_state2 is not None and room_name_state1 == room_name_state2:
                    available_rooms[room_id] = room_name_state1

        room_objects_state1 = {}
        room_objects = self.room_objects[state1]
        for room_id, objects in room_objects.items():
            objects_dict = {}
            for obj in objects:
                objects_dict[obj['objectId']] = obj
            room_objects_state1[room_id] = objects_dict

        room_objects_state2 = {}
        room_objects = self.room_objects[state2]
        for room_id, objects in room_objects.items():
            objects_dict = {}
            for obj in objects:
                objects_dict[obj['objectId']] = obj
            room_objects_state2[room_id] = objects_dict
        
        room_keys = room_objects_state1.keys() & room_objects_state2.keys()

        common_objects = {}
        for room_id in room_keys:
            objects_state1 = room_objects_state1[room_id]
            objects_state2 = room_objects_state2[room_id]
            common_object_ids = objects_state1.keys() & objects_state2.keys()
            for obj_id in common_object_ids:
                if objects_state1[obj_id]['pickupable'] == False:
                    continue
                obj_names = self.object_names[state1][obj_id]
                obj_name_state1, obj_key1 = self._select_name(obj_names, obj_available_keys)
                obj_names = self.object_names[state2][obj_id]
                obj_name_state2, obj_key2 = self._select_name(obj_names, obj_available_keys)
                if obj_name_state1 and obj_name_state2 and obj_name_state1 == obj_name_state2:
                    if room_id not in common_objects:
                        common_objects[room_id] = {}
                    common_objects[room_id][obj_id] = obj_name_state1

        operations = operations_log['operations']
        operations_dict = {}
        for op in operations:
            if op['action'] == "PlaceObjectAtPoint":
                operations_dict[op['object_id']] = op

        # Generate questions for each room
        questions_with_answers = []
        questions_without_answers = []
        
        for room_id, objects in common_objects.items():
            if len(objects) < 4:
                continue
                
            # Get room name for this room
            room_name = available_rooms.get(room_id)
            if not room_name:
                continue
            
            used_objects = set()
            # Generate multiple questions for this room until we run out of objects
            while True:
                available_objects = [obj_id for obj_id in objects.keys() if objects[obj_id] not in used_objects]
                
                if len(available_objects) < 3:
                    break
                
                num_choices = min(random.randint(3, 5), len(available_objects))
                selected_objects = random.sample(available_objects, num_choices)
                
                changed_objects = []
                for obj_id in selected_objects:
                    if obj_id in operations_dict:
                        op = operations_dict[obj_id]
                        changed_objects.append(obj_id)
                
                object_choices = [(objects[obj_id], obj_id) for obj_id in selected_objects]
                object_choices.append((self.no_valid_option, 'none'))
                
                random.shuffle(object_choices)
                choices = [choice[0] for choice in object_choices]
                shuffled_obj_ids = [choice[1] for choice in object_choices]
                
                if changed_objects:
                    correct_answers = [i for i, obj_id in enumerate(shuffled_obj_ids) if obj_id in changed_objects]
                    is_hallucination = False
                else:
                    correct_answers = [i for i, obj_id in enumerate(shuffled_obj_ids) if obj_id == 'none']
                    is_hallucination = True

                related_frames = {state1: set(), state2: set()}
                for obj_id in selected_objects:  # Iterate over all selected objects
                    frames1 = self.get_object_visible_frames(state1, obj_id)
                    frames2 = self.get_object_visible_frames(state2, obj_id)
                    related_frames[state1].update(frames1)
                    related_frames[state2].update(frames2)
                related_frames[state1] = sorted(list(related_frames[state1]))
                related_frames[state2] = sorted(list(related_frames[state2]))
                
                answer_source = []

                question_text = question_template.format(room=room_name, state1=state1, state2=state2)
                question = {
                    'question': question_text,
                    'choices': choices,
                    'correct_answer': correct_answers,
                    'question_type': self.templates['room_position_changes']['question_type'],
                    'category': 'object_changes',
                    'subcategory': 'room_position_changes',
                    'capabilities': self.templates['room_position_changes']['capabilities'],
                    'answer_source': answer_source,
                    'related_frames': related_frames,
                    'hallucination': is_hallucination
                }
                
                # Categorize questions based on whether they have real answers
                if is_hallucination:
                    questions_with_answers.append(question)
                else:
                    questions_without_answers.append(question)
                
                # Mark objects as used
                for obj_id in selected_objects:
                    used_objects.add(objects[obj_id])
        
        # Filter questions based on the requirements
        if len(questions_with_answers) > 0:
            # If there are questions with real answers, limit none-answer questions to not exceed them
            max_none_questions = len(questions_with_answers)
            selected_none_questions = random.sample(questions_without_answers, 
                                                   min(max_none_questions, len(questions_without_answers)//2))
            questions = questions_with_answers + selected_none_questions
        else:
            # If no questions with real answers, keep at most 2 none-answer questions
            questions = random.sample(questions_without_answers, min(5, len(questions_without_answers)//2))
                    
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    

    def _generate_general_attribute_change_questions(self, common_objects: Dict, operations_dict: Dict, state1: str, state2: str) -> List[Dict[str, Any]]:
        """Generate general attribute change questions."""
        template = self.templates['attribute_changed_objects']['question_template']

        questions_with_answers = []
        questions_without_answers = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)
        
        # Collect all objects from all rooms
        all_objects = []
        for room_id, objects in common_objects.items():
            for obj_id, obj_data in objects.items():
                all_objects.append({
                    'id': obj_id,
                    'name': obj_data['name'],
                    'room_id': room_id
                })
        
        if len(all_objects) < 3:
            return []
        
        used_objects = set()
        # Generate multiple questions until we run out of objects
        while True:
            available_objects = [obj for obj in all_objects if obj['name'] not in used_objects]
            
            if len(available_objects) < 3:
                break
            
            num_choices = min(random.randint(3, 5), len(available_objects))
            selected_objects = random.sample(available_objects, num_choices)
            
            changed_objects = []
            for obj in selected_objects:
                if obj['id'] in operations_dict:
                    changed_objects.append(obj)
            
            object_choices = [(obj['name'], obj['id']) for obj in selected_objects]
            object_choices.append((self.no_valid_option, 'none'))
            
            random.shuffle(object_choices)
            choices = [choice[0] for choice in object_choices]
            shuffled_obj_ids = [choice[1] for choice in object_choices]
            
            if changed_objects:
                correct_answers = [i for i, obj_id in enumerate(shuffled_obj_ids) if obj_id in [obj['id'] for obj in changed_objects]]
                is_hallucination = False
            else:
                correct_answers = [i for i, obj_id in enumerate(shuffled_obj_ids) if obj_id == 'none']
                is_hallucination = True
            
            related_frames = {state1: set(), state2: set()}
            for obj in selected_objects:
                frames1 = self.get_object_visible_frames(state1, obj['id'])
                frames2 = self.get_object_visible_frames(state2, obj['id'])
                related_frames[state1].update(frames1)
                related_frames[state2].update(frames2)
            # Convert sets to sorted lists
            related_frames[state1] = sorted(list(related_frames[state1]))
            related_frames[state2] = sorted(list(related_frames[state2]))
            
            answer_source = []

            question_text = template.format(state1=state1, state2=state2)
            question = {
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answers,
                'question_type': self.templates['attribute_changed_objects']['question_type'],
                'category': 'object_changes',
                'subcategory': 'attribute_changed_objects',
                'capabilities': self.templates['attribute_changed_objects']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            }
            
            # Categorize questions based on whether they have real answers
            if is_hallucination:
                questions_with_answers.append(question)
            else:
                questions_without_answers.append(question)
            
            # Mark objects as used
            for obj in selected_objects:
                used_objects.add(obj['name'])
        
        # Filter questions based on the requirements
        if len(questions_with_answers) > 0:
            # If there are questions with real answers, limit none-answer questions to not exceed them
            max_none_questions = len(questions_with_answers)
            selected_none_questions = random.sample(questions_without_answers, 
                                                   min(max_none_questions, len(questions_without_answers)//2))
            questions = questions_with_answers + selected_none_questions
        else:
            # If no questions with real answers, keep at most 2 none-answer questions
            questions = random.sample(questions_without_answers, min(5, len(questions_without_answers)//2))
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    
    def _generate_room_attribute_change_questions(self, common_objects: Dict, available_rooms: Dict, operations_dict: Dict, state1: str, state2: str) -> List[Dict[str, Any]]:
        """Generate room attribute change questions."""
        template = self.templates['room_attribute_changed_objects']['question_template']

        questions_with_answers = []
        questions_without_answers = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)
        
        # Generate questions for each room
        for room_id, objects in common_objects.items():
            # Skip rooms with too few objects
            if len(objects) < 3:
                continue
                
            # Get room name for this room
            room_name = available_rooms.get(room_id)
            if not room_name:
                continue
            
            # Collect objects for this room
            room_objects = []
            for obj_id, obj_data in objects.items():
                room_objects.append({
                    'id': obj_id,
                    'name': obj_data['name'],
                    'room_id': room_id
                })
            
            used_objects = set()
            # Generate multiple questions for this room until we run out of objects
            while True:
                available_objects = [obj for obj in room_objects if obj['name'] not in used_objects]
                
                if len(available_objects) < 3:
                    break
                
                num_choices = min(random.randint(3, 5), len(available_objects))
                selected_objects = random.sample(available_objects, num_choices)
                
                # Check which objects have room attribute changes
                changed_objects = []
                for obj in selected_objects:
                    if obj['id'] in operations_dict:
                        changed_objects.append(obj)
                
                # Create choices with object names and none option
                object_choices = [(obj['name'], obj['id']) for obj in selected_objects]
                object_choices.append((self.no_valid_option, 'none'))
                random.shuffle(object_choices)
                
                choices = [choice[0] for choice in object_choices]
                shuffled_obj_ids = [choice[1] for choice in object_choices]
                
                if changed_objects:
                    correct_answers = [i for i, obj_id in enumerate(shuffled_obj_ids) if obj_id in [obj['id'] for obj in changed_objects]]
                    is_hallucination = False
                else:
                    # Single choice question with none option as correct answer
                    correct_answers = [i for i, obj_id in enumerate(shuffled_obj_ids) if obj_id == 'none']
                    is_hallucination = True
                
                related_frames = {state1: set(), state2: set()}
                for obj in selected_objects:  # Collect visible frames for all selected objects
                    frames1 = self.get_object_visible_frames(state1, obj['id'])
                    frames2 = self.get_object_visible_frames(state2, obj['id'])
                    related_frames[state1].update(frames1)
                    related_frames[state2].update(frames2)
                # Convert sets to sorted lists
                related_frames[state1] = sorted(list(related_frames[state1]))
                related_frames[state2] = sorted(list(related_frames[state2]))
                
                answer_source = []

                question_text = template.format(state1=state1, state2=state2, room=room_name)
                question = {
                    'question': question_text,
                    'choices': choices,
                    'correct_answer': correct_answers,
                    'question_type': self.templates['room_attribute_changed_objects']['question_type'],
                    'category': 'object_changes',
                    'subcategory': 'room_attribute_changed_objects',
                    'capabilities': self.templates['room_attribute_changed_objects']['capabilities'],
                    'answer_source': answer_source,
                    'related_frames': related_frames,
                    'hallucination': is_hallucination
                }
                
                # Categorize questions based on whether they have real answers
                if is_hallucination:
                    questions_with_answers.append(question)
                else:
                    questions_without_answers.append(question)
                
                # Mark objects as used
                for obj in selected_objects:
                    used_objects.add(obj['name'])
        
        # Filter questions based on the requirements
        if len(questions_with_answers) > 0:
            # If there are questions with real answers, limit none-answer questions to not exceed them
            max_none_questions = len(questions_with_answers)
            selected_none_questions = random.sample(questions_without_answers, 
                                                   min(max_none_questions, len(questions_without_answers)//2))
            questions = questions_with_answers + selected_none_questions
        else:
            # If no questions with real answers, keep at most 2 none-answer questions
            questions = random.sample(questions_without_answers, min(5, len(questions_without_answers)//2))
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    
    def _generate_object_attribute_change_questions(self, operations_log: Dict, state1: str, state2: str):
        obj_available_keys = ['type', 'with_contents', 'in_receptacle', 'size', 'position', 'attribute']
        room_available_keys = ['type', 'biggest_room', 'smallest_room', 'biggest_type', 'smallest_type', 'shape']
        
        questions = []

        rooms = self.scene_data['room_static']['room_static_details']
        available_rooms = {}
        for room in rooms:
            room_id = room['room_name']
            
            if room_id in self.room_names[state1]:
                room_names = self.room_names[state1][room_id]
                room_name_state1, room_key1 = self._select_name(room_names, room_available_keys)
                room_names = self.room_names[state2][room_id]
                room_name_state2, room_key2 = self._select_name(room_names, room_available_keys)
                if room_name_state1 is not None and room_name_state2 is not None and room_name_state1 == room_name_state2:
                    available_rooms[room_id] = room_name_state1

        room_objects = self.room_objects[state1]
        room_objects_state1 = {}
        for room_id, objects in room_objects.items():
            objects_dict = {}
            for obj in objects:
                objects_dict[obj['objectId']] = obj
            room_objects_state1[room_id] = objects_dict

        room_objects = self.room_objects[state2]
        room_objects_state2 = {}
        for room_id, objects in room_objects.items():
            objects_dict = {}
            for obj in objects:
                objects_dict[obj['objectId']] = obj
            room_objects_state2[room_id] = objects_dict
        
        room_keys = room_objects_state1.keys() & room_objects_state2.keys()

        # Find common objects in both states
        common_objects = {}
        for room_id in room_keys:
            objects_state1 = room_objects_state1[room_id]
            objects_state2 = room_objects_state2[room_id]
            common_object_ids = objects_state1.keys() & objects_state2.keys()
            for obj_id in common_object_ids:
                obj_names = self.object_names[state1][obj_id]
                obj_name_state1, obj_key1 = self._select_name(obj_names, obj_available_keys)
                obj_names = self.object_names[state2][obj_id]
                obj_name_state2, obj_key2 = self._select_name(obj_names, obj_available_keys)
                if obj_name_state1 and obj_name_state2 and obj_name_state1 == obj_name_state2:
                    if room_id not in common_objects:
                        common_objects[room_id] = {}
                    common_objects[room_id][obj_id] = {
                        'name': obj_name_state1,
                        'obj_state1': objects_state1[obj_id],
                        'obj_state2': objects_state2[obj_id]
                    }

        # Parse operations to understand what changed
        operations = operations_log['operations']
        operations_dict = {}
        for op in operations:
            if op['action'] != "PlaceObjectAtPoint":
                operations_dict[op['object_id']] = op

        # Generate different types of attribute change questions
        questions.extend(self._generate_general_attribute_change_questions(common_objects, operations_dict, state1, state2))
        questions.extend(self._generate_room_attribute_change_questions(common_objects, available_rooms, operations_dict, state1, state2))
        
        return questions


    def _generate_openness_questions(self, obj_names: Dict, state1: str, state2: str, openable_operated_ids: Dict) -> List[Dict[str, Any]]:
        """Generate openness-related questions."""
        question_template = self.templates['openness_state_comparison']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)

        openable_obj_ids = openable_operated_ids['operable']
        openness_changed_obj_ids = openable_operated_ids['operated']
        if len(openness_changed_obj_ids) == 0:
            selected_obj_ids = random.sample(openable_obj_ids, min(5, len(openable_obj_ids)))
        else:
            selected_obj_ids = openness_changed_obj_ids + random.sample(openable_obj_ids, min(len(openness_changed_obj_ids), len(openable_obj_ids)))
        
        # Get object information for both states directly from scene_data
        all_objects_state1 = self.scene_data[state1]['objects_state']['objects_state']
        all_objects_state2 = self.scene_data[state2]['objects_state']['objects_state']

        # Define all possible choice options
        all_choices = [
            "Fully closed in both states",
            "Fully open in both states", 
            "Partially open in both states",
            "Fully closed in the previous state and fully open in the latter state",
            "Fully closed in the previous state and partially open in the latter state",
            "Partially open in the previous state and fully open in the latter state",
            "Partially open in the previous state and fully closed in the latter state",
            "Fully open in the previous state and fully closed in the latter state",
            "Fully open in the previous state and partially open in the latter state"
        ]
        
        for obj_id in selected_obj_ids:
            if obj_id not in obj_names:
                continue
            obj_name = obj_names[obj_id]
            obj_state1 = all_objects_state1[obj_id]
            obj_state2 = all_objects_state2[obj_id]
            
            # Check object's openness state (0=fully closed, 1=fully open, 0-1=partially open)
            openness_state1 = obj_state1['openness']
            openness_state2 = obj_state2['openness']
            
            # Helper function to determine openness category
            def get_openness_category(openness):
                if openness == 0.0:
                    return "fully closed"
                elif openness == 1.0:
                    return "fully open"
                else:
                    return "partially open"
            
            # Get openness categories for both states
            category_state1 = get_openness_category(openness_state1)
            category_state2 = get_openness_category(openness_state2)
            
            # Determine the correct answer based on openness categories
            if category_state1 == "fully closed" and category_state2 == "fully closed":
                correct_choice = "Fully closed in both states"
            elif category_state1 == "fully open" and category_state2 == "fully open":
                correct_choice = "Fully open in both states"
            elif category_state1 == "partially open" and category_state2 == "partially open":
                correct_choice = "Partially open in both states"
            elif category_state1 == "fully closed" and category_state2 == "fully open":
                correct_choice = "Fully closed in the previous state and fully open in the latter state"
            elif category_state1 == "fully closed" and category_state2 == "partially open":
                correct_choice = "Fully closed in the previous state and partially open in the latter state"
            elif category_state1 == "partially open" and category_state2 == "fully open":
                correct_choice = "Partially open in the previous state and fully open in the latter state"
            elif category_state1 == "partially open" and category_state2 == "fully closed":
                correct_choice = "Partially open in the previous state and fully closed in the latter state"
            elif category_state1 == "fully open" and category_state2 == "fully closed":
                correct_choice = "Fully open in the previous state and fully closed in the latter state"
            else:  # category_state1 == "fully open" and category_state2 == "partially open"
                correct_choice = "Fully open in the previous state and partially open in the latter state"
            
            other_choices = [choice for choice in all_choices if choice != correct_choice]
            num_other_choices = random.randint(3, 5)
            selected_other_choices = random.sample(other_choices, min(num_other_choices, len(other_choices)))
            
            choices = [correct_choice] + selected_other_choices
            
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
            correct_answer = [choices.index(correct_choice)]
            
            related_frames = {state1: [], state2: []}
            frames1 = self.get_object_visible_frames(state1, obj_id)
            frames2 = self.get_object_visible_frames(state2, obj_id)
            related_frames[state1] = frames1
            related_frames[state2] = frames2

            answer_source = []
        
            question_text = question_template.format(object_name=obj_name, state1=state1, state2=state2)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['openness_state_comparison']['question_type'],
                'category': 'object_changes',
                'subcategory': 'openness_state_comparison',
                'capabilities': self.templates['openness_state_comparison']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    
    def _generate_toggle_questions(self, obj_names: Dict, state1: str, state2: str, toggleable_operated_ids: Dict) -> List[Dict[str, Any]]:
        """Generate toggle state questions."""
        question_template = self.templates['toggle_state_comparison']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)
        
        toggleable_obj_ids = toggleable_operated_ids['operable']
        toggle_changed_obj_ids = toggleable_operated_ids['operated']
        if len(toggle_changed_obj_ids) == 0:
            selected_obj_ids = random.sample(toggleable_obj_ids, min(5, len(toggleable_obj_ids)))
        else:
            selected_obj_ids = toggle_changed_obj_ids + random.sample(toggleable_obj_ids, min(len(toggle_changed_obj_ids), len(toggleable_obj_ids)))
        
        # Get object information for both states directly from scene_data
        all_objects_state1 = self.scene_data[state1]['objects_state']['objects_state']
        all_objects_state2 = self.scene_data[state2]['objects_state']['objects_state']
        
        # Define all possible choice options
        all_choices = [
            "On in both states",
            "Off in both states", 
            "On in the previous state and off in the latter state",
            "Off in the latter state and on in the previous state"
        ]
        
        for obj_id in selected_obj_ids:
            if obj_id not in obj_names:
                continue
            obj_name = obj_names[obj_id]
            obj_state1 = all_objects_state1[obj_id]
            obj_state2 = all_objects_state2[obj_id]
            
            # Check object's toggle state
            is_toggled_on_state1 = obj_state1['isToggled']
            is_toggled_on_state2 = obj_state2['isToggled']
            
            # Determine the correct answer based on toggle states
            if is_toggled_on_state1 and is_toggled_on_state2:
                correct_choice = "On in both states"
            elif not is_toggled_on_state1 and not is_toggled_on_state2:
                correct_choice = "Off in both states"
            elif is_toggled_on_state1 and not is_toggled_on_state2:
                correct_choice = "On in the previous state and off in the latter state"
            else:  # not is_toggled_on_state1 and is_toggled_on_state2
                correct_choice = "Off in the latter state and on in the previous state"
            
            choices = all_choices.copy()
            
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
            correct_answer = [choices.index(correct_choice)]
            
            related_frames = {state1: [], state2: []}
            frames1 = self.get_object_visible_frames(state1, obj_id)
            frames2 = self.get_object_visible_frames(state2, obj_id)
            related_frames[state1] = frames1
            related_frames[state2] = frames2

            answer_source = []

            question_text = question_template.format(object_name=obj_name, state1=state1, state2=state2)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['toggle_state_comparison']['question_type'],
                'category': 'object_changes',
                'subcategory': 'toggle_state_comparison',
                'capabilities': self.templates['toggle_state_comparison']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    
    def _generate_cleanliness_questions(self, obj_names: Dict, state1: str, state2: str, cleanable_operated_ids: Dict) -> List[Dict[str, Any]]:
        """Generate cleanliness state questions."""
        question_template = self.templates['cleanliness_state_comparison']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)
       
        cleanable_obj_ids = cleanable_operated_ids['operable']
        cleanliness_changed_obj_ids = cleanable_operated_ids['operated']
        if len(cleanliness_changed_obj_ids) == 0:
            selected_obj_ids = random.sample(cleanable_obj_ids, min(5, len(cleanable_obj_ids)))
        else:
            selected_obj_ids = cleanliness_changed_obj_ids + random.sample(cleanable_obj_ids, min(len(cleanliness_changed_obj_ids), len(cleanable_obj_ids)))
        
        # Get object information for both states directly from scene_data
        all_objects_state1 = self.scene_data[state1]['objects_state']['objects_state']
        all_objects_state2 = self.scene_data[state2]['objects_state']['objects_state']
        
        # Define all possible choice options
        all_choices = [
            "Clean in both states",
            "Dirty in both states", 
            "Clean in the previous state and dirty in the latter state",
            "Dirty in the previous state and clean in the latter state"
        ]
        
        for obj_id in selected_obj_ids:
            if obj_id not in obj_names:
                continue
            obj_name = obj_names[obj_id]
            obj_state1 = all_objects_state1[obj_id]
            obj_state2 = all_objects_state2[obj_id]
            
            # Check object's cleanliness state
            is_dirty_state1 = obj_state1['isDirty']
            is_dirty_state2 = obj_state2['isDirty']
            
            # Determine the correct answer based on cleanliness states
            if is_dirty_state1 and is_dirty_state2:
                correct_choice = "Dirty in both states"
            elif not is_dirty_state1 and not is_dirty_state2:
                correct_choice = "Clean in both states"
            elif is_dirty_state1 and not is_dirty_state2:
                correct_choice = "Dirty in the previous state and clean in the latter state"
            else:  # not is_dirty_state1 and is_dirty_state2
                correct_choice = "Clean in the previous state and dirty in the latter state"
            
            choices = all_choices.copy()
            
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
            correct_answer = [choices.index(correct_choice)]
            
            related_frames = {state1: [], state2: []}
            frames1 = self.get_object_visible_frames(state1, obj_id)
            frames2 = self.get_object_visible_frames(state2, obj_id)
            related_frames[state1] = frames1
            related_frames[state2] = frames2

            answer_source = []

            question_text = question_template.format(object_name=obj_name, state1=state1, state2=state2)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['cleanliness_state_comparison']['question_type'],
                'category': 'object_changes',
                'subcategory': 'cleanliness_state_comparison',
                'capabilities': self.templates['cleanliness_state_comparison']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    
    def _generate_cooking_questions(self, obj_names: Dict, state1: str, state2: str, cookable_operated_ids: Dict) -> List[Dict[str, Any]]:
        """Generate cooking state questions."""
        question_template = self.templates['cooking_state_comparison']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)
        
        cookable_obj_ids = cookable_operated_ids['operable']
        cooking_changed_obj_ids = cookable_operated_ids['operated']
        if len(cooking_changed_obj_ids) == 0:
            selected_obj_ids = random.sample(cookable_obj_ids, min(5, len(cookable_obj_ids)))
        else:
            selected_obj_ids = cooking_changed_obj_ids + random.sample(cookable_obj_ids, min(len(cooking_changed_obj_ids), len(cookable_obj_ids)))
        
        # Get object information for both states directly from scene_data
        all_objects_state1 = self.scene_data[state1]['objects_state']['objects_state']
        all_objects_state2 = self.scene_data[state2]['objects_state']['objects_state']
        
        # Define all possible choice options
        all_choices = [
            "Cooked in both states",
            "Uncooked in both states", 
            "Cooked in the previous state and uncooked in the latter state",
            "Uncooked in the previous state and cooked in the latter state"
        ]
        
        for obj_id in selected_obj_ids:
            if obj_id not in obj_names:
                continue
            obj_name = obj_names[obj_id]
            obj_state1 = all_objects_state1[obj_id]
            obj_state2 = all_objects_state2[obj_id]
            
            # Check object's cooking state
            is_cooked_state1 = obj_state1['isCooked']
            is_cooked_state2 = obj_state2['isCooked']
            
            # Determine the correct answer based on cooking states
            if is_cooked_state1 and is_cooked_state2:
                correct_choice = "Cooked in both states"
            elif not is_cooked_state1 and not is_cooked_state2:
                correct_choice = "Uncooked in both states"
            elif is_cooked_state1 and not is_cooked_state2:
                correct_choice = "Cooked in the previous state and uncooked in the latter state"
            else:  # not is_cooked_state1 and is_cooked_state2
                correct_choice = "Uncooked in the previous state and cooked in the latter state"
            
            choices = all_choices.copy()
            
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
            correct_answer = [choices.index(correct_choice)]
            
            related_frames = {state1: [], state2: []}
            frames1 = self.get_object_visible_frames(state1, obj_id)
            frames2 = self.get_object_visible_frames(state2, obj_id)
            related_frames[state1] = frames1
            related_frames[state2] = frames2

            answer_source = []

            question_text = question_template.format(object_name=obj_name, state1=state1, state2=state2)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['cooking_state_comparison']['question_type'],
                'category': 'object_changes',
                'subcategory': 'cooking_state_comparison',
                'capabilities': self.templates['cooking_state_comparison']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    
    def _generate_breaking_questions(self, obj_names: Dict, state1: str, state2: str, breakable_operated_ids: Dict) -> List[Dict[str, Any]]:
        """Generate breaking state questions."""
        question_template = self.templates['breaking_state_comparison']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)
        
        breakable_obj_ids = breakable_operated_ids['operable']
        breaking_changed_obj_ids = breakable_operated_ids['operated']
        if len(breaking_changed_obj_ids) == 0:
            selected_obj_ids = random.sample(breakable_obj_ids, min(5, len(breakable_obj_ids)))
        else:
            selected_obj_ids = breaking_changed_obj_ids + random.sample(breakable_obj_ids, min(len(breaking_changed_obj_ids), len(breakable_obj_ids)))
        
        # Get object information for both states directly from scene_data
        all_objects_state1 = self.scene_data[state1]['objects_state']['objects_state']
        all_objects_state2 = self.scene_data[state2]['objects_state']['objects_state']
        
        # Define all possible choice options
        all_choices = [
            "Intact in both states",
            "Broken in both states", 
            "Intact in the previous state and broken in the latter state",
            "Broken in the previous state and intact in the latter state"
        ]
        
        for obj_id in selected_obj_ids:
            if obj_id not in obj_names:
                continue
            obj_name = obj_names[obj_id]
            obj_state1 = all_objects_state1[obj_id]
            obj_state2 = all_objects_state2[obj_id]
            
            # Check object's breaking state
            is_broken_state1 = obj_state1['isBroken']
            is_broken_state2 = obj_state2['isBroken']
            
            # Determine the correct answer based on breaking states
            if is_broken_state1 and is_broken_state2:
                correct_choice = "Broken in both states"
            elif not is_broken_state1 and not is_broken_state2:
                correct_choice = "Intact in both states"
            elif is_broken_state1 and not is_broken_state2:
                correct_choice = "Broken in the previous state and intact in the latter state"
            else:  # not is_broken_state1 and is_broken_state2
                correct_choice = "Intact in the previous state and broken in the latter state"
            
            choices = all_choices.copy()
            
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
            correct_answer = [choices.index(correct_choice)]
            
            related_frames = {state1: [], state2: []}
            frames1 = self.get_object_visible_frames(state1, obj_id)
            frames2 = self.get_object_visible_frames(state2, obj_id)
            related_frames[state1] = frames1
            related_frames[state2] = frames2

            answer_source = []

            question_text = question_template.format(object_name=obj_name, state1=state1, state2=state2)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['breaking_state_comparison']['question_type'],
                'category': 'object_changes',
                'subcategory': 'breaking_state_comparison',
                'capabilities': self.templates['breaking_state_comparison']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions

    def _generate_slicing_questions(self, obj_names: Dict, state1: str, state2: str, sliceable_operated_ids: Dict) -> List[Dict[str, Any]]:
        """Generate slicing state questions."""
        question_template = self.templates['slicing_state_change']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)
        
        sliceable_obj_ids = sliceable_operated_ids['operable']
        slicing_changed_obj_ids = sliceable_operated_ids['operated']
        if len(slicing_changed_obj_ids) == 0:
            selected_obj_ids = random.sample(sliceable_obj_ids, min(5, len(sliceable_obj_ids)))
        else:
            selected_obj_ids = slicing_changed_obj_ids + random.sample(sliceable_obj_ids, min(len(slicing_changed_obj_ids), len(sliceable_obj_ids)))
        
        # Get object information for both states directly from scene_data
        all_objects_state1 = self.scene_data[state1]['objects_state']['objects_state']
        all_objects_state2 = self.scene_data[state2]['objects_state']['objects_state']
        
        # Define all possible choice options
        all_choices = [
            "Whole in both states",
            "Sliced in both states",
            "Whole in the previous state and sliced in the latter state",
            "Sliced in the previous state and whole in the latter state"
        ]
        
        for obj_id in selected_obj_ids:
            if obj_id not in obj_names:
                continue
            obj_name = obj_names[obj_id]
            obj_state1 = all_objects_state1[obj_id]
            obj_state2 = all_objects_state2[obj_id]
            
            # Check object's slicing state
            is_sliced_state1 = obj_state1['isSliced']
            is_sliced_state2 = obj_state2['isSliced']
            
            # Determine the correct answer based on slicing states
            if is_sliced_state1 and is_sliced_state2:
                correct_choice = "Sliced in both states"
            elif not is_sliced_state1 and not is_sliced_state2:
                correct_choice = "Whole in both states"
            elif not is_sliced_state1 and is_sliced_state2:
                correct_choice = "Whole in the previous state and sliced in the latter state"
            else:  # is_sliced_state1 and not is_sliced_state2
                correct_choice = "Sliced in the previous state and whole in the latter state"
            
            other_choices = [choice for choice in all_choices if choice != correct_choice]
            num_other_choices = random.randint(3, 5)
            selected_other_choices = random.sample(other_choices, min(num_other_choices, len(other_choices)))
            
            choices = [correct_choice] + selected_other_choices

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
            correct_answer = [choices.index(correct_choice)]
    
            related_frames = {state1: [], state2: []}
            frames1 = self.get_object_visible_frames(state1, obj_id)
            frames2 = self.get_object_visible_frames(state2, obj_id)
            related_frames[state1] = frames1
            related_frames[state2] = frames2

            answer_source = []

            question_text = question_template.format(object_name=obj_name, state1=state1, state2=state2)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['slicing_state_change']['question_type'],
                'category': 'object_changes',
                'subcategory': 'slicing_state_change',
                'capabilities': self.templates['slicing_state_change']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    
    def _generate_usage_questions(self, obj_names: Dict, state1: str, state2: str, usable_operated_ids: Dict) -> List[Dict[str, Any]]:
        """Generate usage state questions."""
        question_template = self.templates['usage_state_change']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)
        
        usable_obj_ids = usable_operated_ids['operable']
        usage_changed_obj_ids = usable_operated_ids['operated']
        if len(usage_changed_obj_ids) == 0:
            selected_obj_ids = random.sample(usable_obj_ids, min(5, len(usable_obj_ids)))
        else:
            selected_obj_ids = usage_changed_obj_ids + random.sample(usable_obj_ids, min(len(usage_changed_obj_ids), len(usable_obj_ids)))
        
        # Get object information for both states directly from scene_data
        all_objects_state1 = self.scene_data[state1]['objects_state']['objects_state']
        all_objects_state2 = self.scene_data[state2]['objects_state']['objects_state']
        
        # Define all possible choice options
        all_choices = [
            "Available in both states",
            "Used up in both states",
            "Available in the previous state and used up in the latter state",
            "Used up in the previous state and available in the latter state"
        ]
        
        for obj_id in selected_obj_ids:
            if obj_id not in obj_names:
                continue
            obj_name = obj_names[obj_id]
            obj_state1 = all_objects_state1[obj_id]
            obj_state2 = all_objects_state2[obj_id]
            
            # Check object's usage state
            is_used_up_state1 = obj_state1['isUsedUp']
            is_used_up_state2 = obj_state2['isUsedUp']
            
            # Determine the correct answer based on usage states
            if is_used_up_state1 and is_used_up_state2:
                correct_choice = "Used up in both states"
            elif not is_used_up_state1 and not is_used_up_state2:
                correct_choice = "Available in both states"
            elif not is_used_up_state1 and is_used_up_state2:
                correct_choice = "Available in the previous state and used up in the latter state"
            else:  # is_used_up_state1 and not is_used_up_state2
                correct_choice = "Used up in the previous state and available in the latter state"
            
            other_choices = [choice for choice in all_choices if choice != correct_choice]
            num_other_choices = random.randint(3, 5)
            selected_other_choices = random.sample(other_choices, min(num_other_choices, len(other_choices)))
            
            choices = [correct_choice] + selected_other_choices

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
            correct_answer = [choices.index(correct_choice)]
            
        
            related_frames = {state1: [], state2: []}
            frames1 = self.get_object_visible_frames(state1, obj_id)
            frames2 = self.get_object_visible_frames(state2, obj_id)
            related_frames[state1] = frames1
            related_frames[state2] = frames2

            answer_source = []
            
            question_text = question_template.format(object_name=obj_name, state1=state1, state2=state2)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['usage_state_change']['question_type'],
                'category': 'object_changes',
                'subcategory': 'usage_state_change',
                'capabilities': self.templates['usage_state_change']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    
    def _generate_liquid_questions(self, obj_names: Dict, state1: str, state2: str, fillable_operated_ids: Dict) -> List[Dict[str, Any]]:
        """Generate liquid state questions."""
        question_template = self.templates['liquid_state_comparison']['question_template']
        questions = []

        trajectory1 = self.scene_data[state1]['agent_trajectory']
        trajectory2 = self.scene_data[state2]['agent_trajectory']
        video_len1 = len(trajectory1)
        video_len2 = len(trajectory2)
        video_num1 = self.get_video_num(state1)
        video_num2 = self.get_video_num(state2)
        
        fillable_obj_ids = fillable_operated_ids['operable']
        liquid_changed_obj_ids = fillable_operated_ids['operated']
        if len(liquid_changed_obj_ids) == 0:
            selected_obj_ids = random.sample(fillable_obj_ids, min(5, len(fillable_obj_ids)))
        else:
            selected_obj_ids = liquid_changed_obj_ids + random.sample(fillable_obj_ids, min(len(liquid_changed_obj_ids), len(fillable_obj_ids)))
        
        # Get object information for both states directly from scene_data
        all_objects_state1 = self.scene_data[state1]['objects_state']['objects_state']
        all_objects_state2 = self.scene_data[state2]['objects_state']['objects_state']
        
        # Define all possible choice options
        all_choices = [
            "Empty in both states",
            "Filled with coffee in both states",
            "Filled with wine in both states", 
            "Filled with water in both states",
            "Empty in the previous state and filled with coffee in the latter state",
            "Empty in the previous state and filled with wine in the latter state",
            "Empty in the previous state and filled with water in the latter state",
            "Filled with coffee in the previous state and empty in the latter state",
            "Filled with wine in the previous state and empty in the latter state",
            "Filled with water in the previous state and empty in the latter state",
            "Filled with coffee in the previous state and filled with wine in the latter state",
            "Filled with coffee in the previous state and filled with water in the latter state",
            "Filled with wine in the previous state and filled with coffee in the latter state",
            "Filled with wine in the previous state and filled with water in the latter state",
            "Filled with water in the previous state and filled with coffee in the latter state",
            "Filled with water in the previous state and filled with wine in the latter state"
        ]
        
        for obj_id in selected_obj_ids:
            if obj_id not in obj_names:
                continue
            obj_name = obj_names[obj_id]
            obj_state1 = all_objects_state1[obj_id]
            obj_state2 = all_objects_state2[obj_id]
            
            # Check object's liquid filling state and liquid type
            is_filled_with_liquid_state1 = obj_state1['isFilledWithLiquid']
            is_filled_with_liquid_state2 = obj_state2['isFilledWithLiquid']
            fill_liquid_state1 = obj_state1['fillLiquid'] if is_filled_with_liquid_state1 else ''
            fill_liquid_state2 = obj_state2['fillLiquid'] if is_filled_with_liquid_state2 else ''
            
            # Helper function to get liquid state description
            def get_liquid_state(is_filled, liquid_type):
                if not is_filled or not liquid_type:
                    return "empty"
                return f"filled with {liquid_type}"
            
            # Get liquid states for both states
            liquid_state1 = get_liquid_state(is_filled_with_liquid_state1, fill_liquid_state1)
            liquid_state2 = get_liquid_state(is_filled_with_liquid_state2, fill_liquid_state2)
            
            # Determine the correct answer based on liquid states
            if liquid_state1 == "empty" and liquid_state2 == "empty":
                correct_choice = "Empty in both states"
            elif liquid_state1 == "filled with coffee" and liquid_state2 == "filled with coffee":
                correct_choice = "Filled with coffee in both states"
            elif liquid_state1 == "filled with wine" and liquid_state2 == "filled with wine":
                correct_choice = "Filled with wine in both states"
            elif liquid_state1 == "filled with water" and liquid_state2 == "filled with water":
                correct_choice = "Filled with water in both states"
            elif liquid_state1 == "empty" and liquid_state2 == "filled with coffee":
                correct_choice = "Empty in the previous state and filled with coffee in the latter state"
            elif liquid_state1 == "empty" and liquid_state2 == "filled with wine":
                correct_choice = "Empty in the previous state and filled with wine in the latter state"
            elif liquid_state1 == "empty" and liquid_state2 == "filled with water":
                correct_choice = "Empty in the previous state and filled with water in the latter state"
            elif liquid_state1 == "filled with coffee" and liquid_state2 == "empty":
                correct_choice = "Filled with coffee in the previous state and empty in the latter state"
            elif liquid_state1 == "filled with wine" and liquid_state2 == "empty":
                correct_choice = "Filled with wine in the previous state and empty in the latter state"
            elif liquid_state1 == "filled with water" and liquid_state2 == "empty":
                correct_choice = "Filled with water in the previous state and empty in the latter state"
            elif liquid_state1 == "filled with coffee" and liquid_state2 == "filled with wine":
                correct_choice = "Filled with coffee in the previous state and filled with wine in the latter state"
            elif liquid_state1 == "filled with coffee" and liquid_state2 == "filled with water":
                correct_choice = "Filled with coffee in the previous state and filled with water in the latter state"
            elif liquid_state1 == "filled with wine" and liquid_state2 == "filled with coffee":
                correct_choice = "Filled with wine in the previous state and filled with coffee in the latter state"
            elif liquid_state1 == "filled with wine" and liquid_state2 == "filled with water":
                correct_choice = "Filled with wine in the previous state and filled with water in the latter state"
            elif liquid_state1 == "filled with water" and liquid_state2 == "filled with coffee":
                correct_choice = "Filled with water in the previous state and filled with coffee in the latter state"
            else:  # liquid_state1 == "filled with water" and liquid_state2 == "filled with wine"
                correct_choice = "Filled with water in the previous state and filled with wine in the latter state"
            
            other_choices = [choice for choice in all_choices if choice != correct_choice]
            num_other_choices = random.randint(3, 5)
            selected_other_choices = random.sample(other_choices, min(num_other_choices, len(other_choices)))
            
            choices = [correct_choice] + selected_other_choices

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
            correct_answer = [choices.index(correct_choice)]           

            related_frames = {state1: [], state2: []}
            frames1 = self.get_object_visible_frames(state1, obj_id)
            frames2 = self.get_object_visible_frames(state2, obj_id)
            related_frames[state1] = frames1
            related_frames[state2] = frames2

            answer_source = []
            
            question_text = question_template.format(object_name=obj_name, state1=state1, state2=state2)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['liquid_state_comparison']['question_type'],
                'category': 'object_changes',
                'subcategory': 'liquid_state_comparison',
                'capabilities': self.templates['liquid_state_comparison']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions

    def _generate_specific_attribute_questions(self, operations_log: Dict, state1: str, state2: str) -> List[Dict[str, Any]]:
        obj_available_keys = ['type', 'with_contents', 'in_receptacle', 'size', 'position', 'attribute']
        questions = []

        # Get object information for both states directly from scene_data
        all_objects_state1 = self.scene_data[state1]['objects_state']['objects_state']
        all_objects_state2 = self.scene_data[state2]['objects_state']['objects_state']
        
        # Find common objects in both states and get object names
        common_object_ids = set(all_objects_state1.keys()) & set(all_objects_state2.keys())
        
        common_objects = {}
        for obj_id in common_object_ids:
            if obj_id in self.object_names[state1] and obj_id in self.object_names[state2]:
                obj_names_state1 = self.object_names[state1][obj_id]
                obj_name_state1, obj_key1 = self._select_name(obj_names_state1, obj_available_keys)
                obj_names_state2 = self.object_names[state2][obj_id]
                obj_name_state2, obj_key2 = self._select_name(obj_names_state2, obj_available_keys)
                
                if obj_name_state1 and obj_name_state2 and obj_name_state1 == obj_name_state2:
                    common_objects[obj_id] = obj_name_state1
        
        # Parse operation log and remove objects with PlaceObjectAtPoint action
        operations = operations_log['operations']
        place_object_ids = set()
        operations_dict = {}
        # Filter objects with changeable attributes and classify them
        changeable_attributes = {
            'openable': {'operable': [], 'operated': []},
            'sliceable': {'operable': [], 'operated': []},
            'breakable': {'operable': [], 'operated': []},
            'toggleable': {'operable': [], 'operated': []},
            'dirtyable': {'operable': [], 'operated': []},
            'cookable': {'operable': [], 'operated': []},
            'canFillWithLiquid': {'operable': [], 'operated': []},
            'canBeUsedUp': {'operable': [], 'operated': []}
        }
        for op in operations:
            if op['object_id'] not in common_objects:
                continue
            if op['action'] == "PlaceObjectAtPoint":
                place_object_ids.add(op['object_id'])
            else:
                operations_dict[op['object_id']] = op
                action = op['action']
                if action == 'CloseObject' or action == 'OpenObject':
                    changeable_attributes['openable']['operated'].append(op['object_id'])
                elif action == 'SliceObject':
                    changeable_attributes['sliceable']['operated'].append(op['object_id'])
                elif action == 'BreakObject':
                    changeable_attributes['breakable']['operated'].append(op['object_id'])
                elif action == 'ToggleObjectOff' or action == 'ToggleObjectOn':
                    changeable_attributes['toggleable']['operated'].append(op['object_id'])
                elif action == 'DirtyObject' or action == 'CleanObject':
                    changeable_attributes['dirtyable']['operated'].append(op['object_id'])
                elif action == 'CookObject':
                    changeable_attributes['cookable']['operated'].append(op['object_id'])
                elif action == 'FillObjectWithLiquid' or action == 'EmptyLiquidFromObject':
                    changeable_attributes['canFillWithLiquid']['operated'].append(op['object_id'])
                elif action == 'UseUpObject':
                    changeable_attributes['canBeUsedUp']['operated'].append(op['object_id'])
            
        # Iterate through all common objects and check their attributes
        for obj_id in common_objects.keys():
            obj_state1 = all_objects_state1[obj_id]
            
            # Check if object has changeable attributes and classify as operable or operated
            for attr_name in changeable_attributes.keys():
                if obj_state1.get(attr_name, False):
                    
                    # Check if this object was actually operated on
                    attr_operated_ids = changeable_attributes[attr_name]['operated']
                    if obj_id not in attr_operated_ids:
                        changeable_attributes[attr_name]['operable'].append(obj_id)

        # Generate questions for each attribute change
        if changeable_attributes['openable']['operable'] or changeable_attributes['openable']['operated']:
            questions.extend(self._generate_openness_questions(
                common_objects, state1, state2, changeable_attributes['openable']
            ))
        
        if changeable_attributes['toggleable']['operable'] or changeable_attributes['toggleable']['operated']:
            questions.extend(self._generate_toggle_questions(
                common_objects, state1, state2, changeable_attributes['toggleable']
            ))
        
        if changeable_attributes['dirtyable']['operable'] or changeable_attributes['dirtyable']['operated']:
            questions.extend(self._generate_cleanliness_questions(
                common_objects, state1, state2, changeable_attributes['dirtyable']
            ))
        
        if changeable_attributes['cookable']['operable'] or changeable_attributes['cookable']['operated']:
            questions.extend(self._generate_cooking_questions(
                common_objects, state1, state2, changeable_attributes['cookable']
            ))
        
        if changeable_attributes['breakable']['operable'] or changeable_attributes['breakable']['operated']:
            questions.extend(self._generate_breaking_questions(
                common_objects, state1, state2, changeable_attributes['breakable']
            ))
        
        if changeable_attributes['sliceable']['operable'] or changeable_attributes['sliceable']['operated']:
            questions.extend(self._generate_slicing_questions(
                common_objects, state1, state2, changeable_attributes['sliceable']
            ))
        
        if changeable_attributes['canBeUsedUp']['operable'] or changeable_attributes['canBeUsedUp']['operated']:
            questions.extend(self._generate_usage_questions(
                common_objects, state1, state2, changeable_attributes['canBeUsedUp']
            ))

        if changeable_attributes['canFillWithLiquid']['operable'] or changeable_attributes['canFillWithLiquid']['operated']:
            questions.extend(self._generate_liquid_questions(
                common_objects, state1, state2, changeable_attributes['canFillWithLiquid']
            ))
        
        return questions
