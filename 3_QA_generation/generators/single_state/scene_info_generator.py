import os
import json
import copy
import random
import inspect
from typing import Dict, List, Any
from ..base_generator import BaseGenerator


class SceneInfoGenerator(BaseGenerator):
    
    def __init__(self, scene_path: str):
        super().__init__(scene_path)

        template_path = os.path.join(os.path.dirname(__file__), '../../templates/single_state/scene_info_template.json')
        with open(template_path, 'r', encoding='utf-8') as f:
            self.templates = json.load(f)['scene_info']
        
    def generate_questions(self) -> List[Dict[str, Any]]:
        questions = []
        
        if 'room_static' not in self.scene_data or 'room_static_details' not in self.scene_data['room_static']:
            return questions

        questions.extend(self._generate_scene_area_questions())
        questions.extend(self._generate_room_area_questions())
        questions.extend(self._generate_room_type_questions())
        questions.extend(self._generate_room_function_questions())
        questions.extend(self._generate_room_shape_questions())
        questions.extend(self._generate_scene_room_types_questions())
        questions.extend(self._generate_room_count_questions())
        questions.extend(self._generate_room_connectivity_questions())
        
        return questions
    
    def _generate_scene_area_questions(self) -> List[Dict[str, Any]]:
        questions = []

        trajectory = self.scene_data['state_0']['agent_trajectory']
        len_video = len(trajectory)

        scene_area = self.scene_data['room_static']['total_area']

        num_choices = random.randint(4, 6)
        range_float = random.random()
        area_range = [scene_area * range_float, scene_area * (1 + range_float)]
        choices, correct_choice = self.create_range_choices(area_range, scene_area, 'm²', num_choices)

        if random.random() < 1/3:
            is_hallucination = True
            correct_answer = self.no_valid_option
            choices.pop(correct_choice)
            choices.append(correct_answer)
        else:
            is_hallucination = False
            correct_answer = copy.copy(choices[correct_choice])
            candidate_remove_ids = [i for i, c in enumerate(choices) if i != correct_choice]
            if candidate_remove_ids:
                remove_id = random.choice(candidate_remove_ids)
                choices.pop(remove_id)
            choices.append(self.no_valid_option)

        random.shuffle(choices)
        correct_choice = choices.index(correct_answer)
        
        # Build answer source with step-by-step room analysis
        rooms = self.scene_data['room_static']['room_static_details']
        answer_source = [
            {'type': 'text', 'content': f'To determine the total area of this scene, we need to analyze each room individually and then sum up their areas. This scene contains {len(rooms)} rooms in total. The following videos show the content of each room. The videos include some objects with labeled dimensions that can help you estimate the room scale. However, these labeled dimensions are approximate and you should use them as reference points to estimate the actual room dimensions. It is necessary to extract the relevant video segments for each room from the full video. '}
        ]
        related_frames = {"state_0": set()}
        
        for i, room in enumerate(rooms):
            clip, selected_obj = self.select_proper_room_frames_with_objects('state_0', room)
            start_frame = clip[0]
            end_frame = clip[-1]
            # clip_data = self.get_one_clip_data(clip, 'state_0', objs = selected_obj, rooms = True, room_area = True)

            # room_area = room['area']
            
            # # Add text description for this room
            # answer_source.append({'type': 'text', 
            #     'content': f'Based on the labeled objects in the video, you need to estimate the area of this room which is {room_area:.2f} m². The video clip is extracted from frames {start_frame} to {end_frame} of the first video with a total length of {len_video} frames:'
            # })
            
            # # Add video clip for this room
            # answer_source.append({'type': 'video', 'content': clip_data, 'state': 'state_0'})

            for frame_id in range(start_frame, end_frame + 1):
                related_frames['state_0'].add(frame_id)
        
        related_frames['state_0'] = sorted(list(related_frames['state_0']))
        
        question_text = self.templates['scene_area']['question_template']
        questions.append({
            'question': question_text,
            'choices': choices,
            'correct_answer': correct_choice,
            'question_type': self.templates['scene_area']['question_type'],
            'category': 'scene_info',
            'subcategory': 'scene_area',
            'capabilities': self.templates['scene_area']['capabilities'],
            'answer_source': answer_source,
            'related_frames': related_frames,
            'hallucination': is_hallucination
        })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    
    def _generate_room_area_questions(self) -> List[Dict[str, Any]]:
        available_keys = ['type', 'biggest_room', 'smallest_room', 'biggest_type', 'smallest_type', 'shape']
        question_template = self.templates['room_area']['question_template']
        questions = []

        trajectory = self.scene_data['state_0']['agent_trajectory']
        len_video = len(trajectory)

        rooms = self.scene_data['room_static']['room_static_details']
        
        for room in rooms:
            area = room['area']
            candidate_room_names = self.room_names['state_0'][room['room_name']]
            room_name, room_key = self._select_name(candidate_room_names, available_keys, False)
            if room_name is None:
                continue
            
            num_choices = random.randint(4, 6)
            range_float = random.random()
            area_range = [area * range_float, area * (1 + range_float)]
            choices, correct_choice = self.create_range_choices(area_range, area, 'm²', num_choices)
            if not choices:
                continue

            if random.random() < 1/3:
                is_hallucination = True
                correct_answer = self.no_valid_option
                choices.pop(correct_choice)
                choices.append(correct_answer)
            else:
                is_hallucination = False
                correct_answer = copy.copy(choices[correct_choice])
                candidate_remove_ids = [i for i, c in enumerate(choices) if i != correct_choice]
                if candidate_remove_ids:
                    remove_id = random.choice(candidate_remove_ids)
                    choices.pop(remove_id)
                choices.append(self.no_valid_option)

            random.shuffle(choices)
            correct_choice = choices.index(correct_answer)
            
            answer_source = [] ###
            clip, selected_obj_ids = self.select_proper_room_frames_with_objects('state_0', room)
            start_frame = clip[0]
            end_frame = clip[-1]
            # clip_data = self.get_one_clip_data(clip, 'state_0', objs = selected_obj_ids, rooms = True, room_area = True)
            
            # answer_source = [
            #     {'type': 'text', 'content': f'To determine the area of the specific room mentioned in the question, you need to first identify which room it is. The video includes some objects with labeled dimensions that can help you estimate the room scale. However, these labeled dimensions are approximate and you should use them as reference points to estimate the actual room dimensions. It is necessary to extract the relevant video segment for this room from the full video. There are {len_video} frames in the video of the first state.'},
            #     {'type': 'text', 'content': f'Based on the labeled objects in the video, you need to estimate the area of this room. The video clip is extracted from frames {start_frame} to {end_frame} of the first video:'},
            #     {'type': 'video', 'content': clip_data, 'state': 'state_0'}
            # ]

            related_frames = {"state_0": clip}
            
            question_text = question_template.format(room_name=room_name)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_choice,
                'question_type': self.templates['room_area']['question_type'],
                'category': 'scene_info',
                'subcategory': 'room_area',
                'capabilities': self.templates['room_area']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    
    def _generate_room_type_questions(self) -> List[Dict[str, Any]]:
        available_keys = ['biggest_room', 'smallest_room', 'shape']
        question_template = self.templates['room_type']['question_template']
        questions = []

        trajectory = self.scene_data['state_0']['agent_trajectory']
        len_video = len(trajectory)

        rooms = self.scene_data['room_static']['room_static_details']
        for room in rooms:
            room_type = room['room_type']
            candidate_room_names = self.room_names['state_0'][room['room_name']]
            room_name, room_key = self._select_name(candidate_room_names, available_keys, False)
            if room_name:
                is_hallucination = random.random() < 1/3
                
                other_types = list(self.room_types.keys())
                random.shuffle(other_types)
                num_choice = random.randint(4,6)
                
                if is_hallucination:
                    choices = []
                    for t in other_types:
                        if t != room_type and t not in choices and len(choices) < num_choice - 1:
                            choices.append(t)
                    choices.append(self.no_valid_option)
                    random.shuffle(choices)
                    correct_answer = choices.index(self.no_valid_option)
                else:
                    choices = [room_type, self.no_valid_option]
                    for t in other_types:
                        if t not in choices and len(choices) < num_choice:
                            choices.append(t)
                    random.shuffle(choices)
                    correct_answer = choices.index(room_type)
                
                answer_source = [] ###
                clip = self.select_proper_room_frames('state_0', room)
                start_frame = clip[0]
                end_frame = clip[-1]
                # clip_data = self.get_one_clip_data(clip, 'state_0')
                # answer_source = [
                #     {'type': 'text', 'content': f'To determine the type of the specific room mentioned in the question, you need to first identify which room it is. You can examine the distinctive objects within the room to infer the type of the room. It is necessary to extract the relevant video segment for this room from the full video. There are {len_video} frames in the video of the first state.'},
                #     {'type': 'text', 'content': f'The room type is {room_type} and the video clip is extracted from frames {start_frame} to {end_frame} of the first video:'},
                #     {'type': 'video', 'content': clip_data, 'state': 'state_0'}
                # ]

                related_frames = {"state_0": clip}
                
                question_text = question_template.format(room_name=room_name)
                questions.append({
                    'question': question_text,
                    'choices': choices,
                    'correct_answer': correct_answer,
                    'question_type': self.templates['room_type']['question_type'],
                    'category': 'scene_info',
                    'subcategory': 'room_type',
                    'capabilities': self.templates['room_type']['capabilities'],
                    'answer_source': answer_source,
                    'related_frames': related_frames,
                    'hallucination': is_hallucination
                })
            ###
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    
    def _generate_room_function_questions(self) -> List[Dict[str, Any]]:
        available_keys = ['biggest_room', 'smallest_room', 'shape']
        
        question_template = self.templates['room_function']['question_template']
        questions = []

        trajectory = self.scene_data['state_0']['agent_trajectory']
        len_video = len(trajectory)
        
        rooms = self.scene_data['room_static']['room_static_details']
        for room in rooms:
            room_type = room['room_type']
            
            if room_type in self.room_types:
                correct_function = self.room_types[room_type]
                candidate_room_names = self.room_names['state_0'][room['room_name']]
                room_name, room_key = self._select_name(candidate_room_names, available_keys, False)
                if not room_name:
                    continue

                is_hallucination = random.random() < 1/3
                
                choice_num = random.randint(4,6)
                all_functions = list(self.room_types.values())
                
                if is_hallucination:
                    choices = []
                    for func in all_functions:
                        if func != correct_function and func not in choices and len(choices) < choice_num - 1:
                            choices.append(func)
                    choices.append(self.no_valid_option)
                    random.shuffle(choices)
                    correct_answer = choices.index(self.no_valid_option)
                else:
                    choices = [correct_function, self.no_valid_option]
                    for func in all_functions:
                        if func not in choices and len(choices) < choice_num:
                            choices.append(func)
                    random.shuffle(choices)
                    correct_answer = choices.index(correct_function)
                
                answer_source = [] ###
                clip = self.select_proper_room_frames('state_0', room)
                start_frame = clip[0]
                end_frame = clip[-1]
                # clip_data = self.get_one_clip_data(clip, 'state_0')
                # answer_source = [
                #     {'type': 'text', 'content': f'To determine the function of the specific room mentioned in the question, you need to first identify which room it is. You can examine the distinctive objects within the room to infer the type and function of the room. It is necessary to extract the relevant video segment for this room from the full video. There are {len_video} frames in the video of the first state.'},
                #     {'type': 'text', 'content': f'The room function is {correct_function} and the video clip is extracted from frames {start_frame} to {end_frame} of the first video:'},
                #     {'type': 'video', 'content': clip_data, 'state': 'state_0'}
                # ]

                related_frames = {"state_0": clip}
                
                question_text = question_template.format(room_name=room_name)
                questions.append({
                    'question': question_text,
                    'choices': choices,
                    'correct_answer': correct_answer,
                    'question_type': self.templates['room_function']['question_type'],
                    'category': 'scene_info',
                    'subcategory': 'room_function',
                    'capabilities': self.templates['room_function']['capabilities'],
                    'answer_source': answer_source,
                    'related_frames': related_frames,
                    'hallucination': is_hallucination
                })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    
    def _generate_room_shape_questions(self) -> List[Dict[str, Any]]:
        available_keys = ['type', 'biggest_room', 'smallest_room', 'biggest_type', 'smallest_type']
        
        question_template = self.templates['room_shape']['question_template']
        questions = []

        trajectory = self.scene_data['state_0']['agent_trajectory']
        len_video = len(trajectory)
        
        rooms = self.scene_data['room_static']['room_static_details']
        for room in rooms:
            correct_answer = room['shape']
            candidate_room_names = self.room_names['state_0'][room['room_name']]
            room_name, room_key = self._select_name(candidate_room_names, available_keys, False)
            if room_name:
                is_hallucination = random.random() < 1/3
                
                num_choices = random.randint(4, 6)
                available_shapes = [shape for shape in self.room_shape_options if shape != correct_answer]
                
                if is_hallucination:
                    if len(available_shapes) >= num_choices - 1:
                        choices = random.sample(available_shapes, num_choices - 1)
                    else:
                        choices = available_shapes.copy()
                    choices.append(self.no_valid_option)
                    correct_answer = self.no_valid_option
                else:
                    if len(available_shapes) >= num_choices - 2:
                        distractor_choices = random.sample(available_shapes, num_choices - 1)
                    else:
                        distractor_choices = available_shapes
                    
                    choices = distractor_choices + [correct_answer, self.no_valid_option]
                random.shuffle(choices)
                correct_choice = choices.index(correct_answer)
                
                answer_source = [] ###
                clip = self.select_proper_room_frames('state_0', room)
                start_frame = clip[0]
                end_frame = clip[-1]
                # clip_data = self.get_one_clip_data(clip, 'state_0')
                
                # answer_source = [
                #     {'type': 'text', 'content': f'To determine the shape of the specific room mentioned in the question, you need to first identify which room it is. You can pay attention to the lines where the walls meet the floor, or the lines where the floor color changes, and then estimate the shape of the room based on the shape formed by these lines. It is necessary to extract the relevant video segment for this room from the full video. There are {len_video} frames in the video of the first state.'},
                #     {'type': 'text', 'content': f'The room shape is {correct_answer} and the video clip is extracted from frames {start_frame} to {end_frame} of the first video:'},
                #     {'type': 'video', 'content': clip_data, 'state': 'state_0'}
                # ]

                related_frames = {"state_0": clip}
                
                question_text = question_template.format(room_name=room_name)
                questions.append({
                    'question': question_text,
                    'choices': choices,
                    'correct_answer': correct_choice,
                    'question_type': self.templates['room_shape']['question_type'],
                    'category': 'scene_info',
                    'subcategory': 'room_shape',
                    'capabilities': self.templates['room_shape']['capabilities'],
                    'answer_source': answer_source,
                    'related_frames': related_frames,
                    'hallucination': is_hallucination
                })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    
    def _generate_scene_room_types_questions(self) -> List[Dict[str, Any]]:
        question_text = self.templates['scene_room_types']['question_template']
        questions = []
        
        room_types = list(self.scene_data['room_static']['room_type_distribution'].keys())
        random.shuffle(room_types)
        all_possible_types = list(self.room_types.keys())
        random.shuffle(all_possible_types)

        choice_num = random.randint(4, 6)

        is_hallucination = random.random() < 1/3
        
        if is_hallucination:
            choices = []
            for t in all_possible_types:
                if t not in room_types and len(choices) < choice_num - 1:
                    choices.append(t)
            choices.append(self.no_valid_option)
            random.shuffle(choices)
            correct_answers = choices.index(self.no_valid_option)
        else:
            correct_choice_num = random.randint(1, len(room_types) + 1)
            correct_choices = room_types[:correct_choice_num]
            incorrect_candidates = [t for t in all_possible_types if t not in room_types]
            incorrect_choices = [incorrect_candidates[i] for i in range(min(choice_num - correct_choice_num, len(incorrect_candidates)))]
            choices = correct_choices + incorrect_choices + [self.no_valid_option]
            random.shuffle(choices)
            correct_answers = [i for i, choice in enumerate(choices) if choice in room_types]

        related_frames = {"state_0": []}
        
        answer_source = [] ###
        # len_video = len(self.scene_data['state_0']['agent_trajectory'])
        # rooms = self.scene_data['room_static']['room_static_details']
        
        # answer_source = [
        #     {'type': 'text', 'content': f'To identify all the different types of rooms that appear in the entire scene, you need to examine each room and identify their types based on the distinctive objects within them. This scene contains {len(rooms)} rooms in total. It is necessary to extract the relevant video segments for all rooms from the full video. There are {len_video} frames in the video of the first state.'}
        # ]
        # for room in rooms:
        #     clip = self.select_proper_room_frames_with_most_visible_objects('state_0', room)
        #     clip_data = self.get_one_clip_data(clip, 'state_0')
        #     start_frame = clip_data['frames'][0]['frame_id']
        #     end_frame = clip_data['frames'][-1]['frame_id']
        #     room_type = room['room_type']

        #     answer_source.append({'type': 'text', 'content': f"The following video shows a {room_type}. The video clip is extracted from frames {start_frame} to {end_frame} of the first video."})
        #     answer_source.append({'type': 'video', 'content': clip_data, 'state': 'state_0'})

        #     for frame_id in range(start_frame, end_frame + 1):
        #         related_frames['state_0'].add(frame_id)
        # answer_source.append({'type': 'text', 'content': f'By examining all the rooms, we can conclude that the scene contains the following room types: {", ".join(room_types)}.'})
        
        # related_frames['state_0'] = sorted(list(related_frames['state_0']))

        questions.append({
            'question': question_text,
            'choices': choices,
            'correct_answer': correct_answers,
            'question_type': self.templates['scene_room_types']['question_type'],
            'category': 'scene_info',
            'subcategory': 'scene_room_types',
            'capabilities': self.templates['scene_room_types']['capabilities'],
            'answer_source': answer_source,
            'related_frames': related_frames,
            'hallucination': is_hallucination
        })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    
    def _generate_room_count_questions(self) -> List[Dict[str, Any]]:
        questions = []
        question_template = self.templates['room_type_count']['question_template']
        
        trajectory = self.scene_data['state_0']['agent_trajectory']
        len_video = len(trajectory)
        rooms = self.scene_data['room_static']['room_static_details']
        room_clips_by_type = {}
        for room in rooms:
            room_type = room['room_type']
            if room_type not in room_clips_by_type:
                room_clips_by_type[room_type] = []
            
            clip = self.select_proper_room_frames_with_most_visible_objects('state_0', room)
            clip_data = self.get_one_clip_data(clip, 'state_0')
            room_clips_by_type[room_type].append(clip_data)
        
        room_type_counts = self.scene_data['room_static']['room_type_distribution']
        for room_type, count in room_type_counts.items():
            is_hallucination = random.random() < 1/3
            
            if is_hallucination:
                choices = [self.no_valid_option]
                num_choices = random.randint(4, 6)
                
                max_scene_rooms = sum(room_type_counts.values())
                all_possible_numerical_options = list(set([str(i) for i in range(max(5, max_scene_rooms + 3))]))
                random.shuffle(all_possible_numerical_options)

                for c in all_possible_numerical_options:
                    if c != str(count) and c not in choices and len(choices) < num_choices:
                        choices.append(c)
                
                random.shuffle(choices)
                correct_answer = choices.index(self.no_valid_option)
            else:
                choices = [count]
                potential_distractors = set()
                for i in range(max(0, count - 5), count + 6):
                    if i != count:
                        potential_distractors.add(i)

                potential_distractors_list = list(potential_distractors)
                random.shuffle(potential_distractors_list)
                num_distractors_to_add = random.randint(3, min(5, len(potential_distractors_list)))
                for i in range(num_distractors_to_add):
                    choices.append(potential_distractors_list[i])
                choices = [str(x) for x in choices]
                choices.append(self.no_valid_option)
                random.shuffle(choices)
                correct_answer = choices.index(str(count))

            related_frames = {"state_0": []}

            answer_source = [] ###
            # clip_datas = room_clips_by_type[room_type]
            # answer_source = [
            #     {'type': 'text', 'content': f'To count the number of {room_type} rooms in the scene, you need to identify and count all rooms of this specific type. It is necessary to extract the relevant video segments for all {room_type} rooms from the full video. There are {len_video} frames in the video of the first state.'},
            #     {'type': 'text', 'content': f'There are {count} {room_type} room(s) in the scene. The video clips show all {room_type} rooms:'},
            # ]
            # for clip_data in clip_datas:
            #     start_frame = clip_data['frames'][0]['frame_id']
            #     end_frame = clip_data['frames'][-1]['frame_id']
            #     answer_source.append({'type': 'text', 'content': f'The video clip is extracted from frames {start_frame} to {end_frame} of the first video:'})
            #     answer_source.append({'type': 'video', 'content': clip_data, 'state': 'state_0'})

            #     for frame_id in range(start_frame, end_frame + 1):
            #         related_frames['state_0'].add(frame_id)
            
            # related_frames['state_0'] = sorted(list(related_frames['state_0']))

            question_text = question_template.format(room_type=room_type)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['room_type_count']['question_type'],
                'category': 'scene_info',
                'subcategory': 'room_type_count',
                'capabilities': self.templates['room_type_count']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions

    def _generate_room_connectivity_questions(self) -> List[Dict[str, Any]]:
        available_keys = ['type', 'biggest_room', 'smallest_room', 'biggest_type', 'smallest_type', 'shape', 'unique_object']
        question_text = self.templates['room_connectivity_pair']['question_template']
        questions = []

        trajectory = self.scene_data['state_0']['agent_trajectory']
        len_video = len(trajectory)

        room_names = {}
        room_dict = {}
        room_id2name = {}
        rooms = self.scene_data['room_static']['room_static_details']
        for room in rooms:
            candidate_room_names = self.room_names['state_0'][room['room_name']]
            selected_name, selected_key = self._select_name(candidate_room_names, available_keys, False)
            if selected_name:
                room_names[room['room_name']] = selected_name
                room_dict[room['room_name']] = room
                room_id2name[room['room_id']] = room['room_name']

        valid_room_ids = [room_id for room_id, names in room_names.items() if names]
        
        if len(valid_room_ids) < 2:
            return []
        
        all_possible_id_pairs = set()
        for i in range(len(valid_room_ids)):
            for j in range(i + 1, len(valid_room_ids)):
                room_id_1 = valid_room_ids[i]
                room_id_2 = valid_room_ids[j]
                pair = tuple(sorted((room_id_1, room_id_2)))
                all_possible_id_pairs.add(pair)

        connected_room_id_pairs = set()
        for room_id in valid_room_ids:
            room_data = room_dict[room_id]
            connected_room_ids = room_data['connected_rooms']
            for connected_room_id in connected_room_ids:
                if connected_room_id in room_id2name.keys():
                    pair = tuple(sorted((room_id, room_id2name[connected_room_id])))
                    connected_room_id_pairs.add(pair)
                
        # related_informations = []
        # for room_id_pair in connected_room_id_pairs:
        #     room_id_1, room_id_2 = room_id_pair
        #     for room in rooms:
        #         if room['room_name'] == room_id_1:
        #             room_1 = room
        #         if room['room_name'] == room_id_2:
        #             room_2 = room
        #     clip = self.select_proper_connected_room_frames('state_0', (room_1, room_2))
        #     clip_data = self.get_one_clip_data(clip, 'state_0', rooms = True)
        #     related_informations.append(clip_data)
        
        used_pairs = set()
        num = 0
        while True:
            available_all_id_pairs = [p for p in all_possible_id_pairs if p not in used_pairs]
            if len(available_all_id_pairs) < 4:
                break
            
            is_hallucination = random.random() < 1/3
            
            num_choices = random.randint(max(1, min(4, len(available_all_id_pairs) + 1)), min(6, len(available_all_id_pairs) + 1))
            num_room_pairs_to_select = num_choices - 1

            selected_pairs = []
            if num_room_pairs_to_select > 0:
                if is_hallucination:
                    incorrect_available_pairs = [p for p in available_all_id_pairs if p not in connected_room_id_pairs]
                    
                    if len(incorrect_available_pairs) < num_room_pairs_to_select:
                        selected_pairs.extend(incorrect_available_pairs)
                        remaining_to_select = num_room_pairs_to_select - len(selected_pairs)
                        if remaining_to_select > 0:
                            other_available_pairs = [p for p in available_all_id_pairs if p not in selected_pairs]
                            selected_pairs.extend(random.sample(other_available_pairs, min(remaining_to_select, len(other_available_pairs))))
                    else:
                        selected_pairs.extend(random.sample(incorrect_available_pairs, num_room_pairs_to_select))
                else:
                    connected_pairs_in_available = [p for p in available_all_id_pairs if p in connected_room_id_pairs]
                    non_connected_pairs_in_available = [p for p in available_all_id_pairs if p not in connected_room_id_pairs]

                    if connected_pairs_in_available and num_room_pairs_to_select > 0:
                        # Select one connected pair
                        chosen_connected_pair = random.choice(connected_pairs_in_available)
                        selected_pairs.append(chosen_connected_pair)
                        num_room_pairs_to_select -= 1
                        # Remove it from its pool to avoid re-selection
                        connected_pairs_in_available.remove(chosen_connected_pair)

                    # Fill the rest of the slots with a mix of remaining connected and non-connected pairs
                    all_remaining_available = connected_pairs_in_available + non_connected_pairs_in_available
                    random.shuffle(all_remaining_available) # Shuffle to mix them
                    
                    selected_pairs.extend(random.sample(all_remaining_available, min(num_room_pairs_to_select, len(all_remaining_available))))
            
            for pair in selected_pairs:
                used_pairs.add(pair)
            
            choices = [f"{room_names[p[0]]} -- {room_names[p[1]]}" for p in selected_pairs]
            choices.append(self.no_valid_option) # Now choices has num_room_pairs_to_select + 1 elements, which is num_choices
            random.shuffle(choices)
            
            if is_hallucination:
                correct_answers = [choices.index(self.no_valid_option)]
                hallucination = True
            else:
                # In non-hallucination mode, correct answers are the connected pairs among selected_pairs
                correct_choices_pairs = [p for p in selected_pairs if p in connected_room_id_pairs]
                correct_choice_names = [f"{room_names[p[0]]} -- {room_names[p[1]]}" for p in correct_choices_pairs]
                
                # If no correct choices were selected among the room pairs, then self.no_valid_option is the correct answer
                if not correct_choice_names:
                    correct_answers = [choices.index(self.no_valid_option)]
                    hallucination = True
                else:
                    correct_answers = [i for i, p in enumerate(choices) if p in correct_choice_names]
                    hallucination = False

            related_frames = {"state_0": []}
            
            answer_source = [] ###
            # answer_source = [
            #     {'type': 'text', 'content': f'To determine room connectivity, you need to identify which rooms are directly connected to each other. It is necessary to extract the relevant video segments from the full video that show transitions between rooms. There are {len_video} frames in the video of the first state.'},
            #     {'type': 'text', 'content': 'The following video clips show room transitions and connectivity:'}
            # ]
            # for clip_data in related_informations:
            #     all_room_ids = set(clip_data['rooms']['visit_order'])
            #     all_room_names = [room_names[room_id] for room_id in all_room_ids]
            #     if len(all_room_names) == 0:
            #         import pdb;pdb.set_trace()
            #     start_frame = clip_data['frames'][0]['frame_id']
            #     end_frame = clip_data['frames'][-1]['frame_id']
            #     answer_source.append({'type': 'text', 'content': f'The video clip is extracted from frames {start_frame} to {end_frame} of the first video and it shows the connectivity between {all_room_names[0]} and {all_room_names[1]}:'})
            #     answer_source.append({'type': 'video', 'content': clip_data, 'state': 'state_0'})

            #     for frame_id in range(start_frame, end_frame + 1):
            #         related_frames['state_0'].add(frame_id)
            # related_frames['state_0'] = sorted(list(related_frames['state_0']))

            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answers,
                'question_type': self.templates['room_connectivity_pair']['question_type'],
                'category': 'scene_info',
                'subcategory': 'room_connectivity_pair',
                'capabilities': self.templates['room_connectivity_pair']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': hallucination
            })

            num += 1
            if num > 10:
                break
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
 