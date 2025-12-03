import os
import json
import copy
import random
import inspect
from typing import Dict, List, Any

from ..base_generator import BaseGenerator

class ObjectInfoGenerator(BaseGenerator):
    
    def __init__(self, scene_path: str):
        super().__init__(scene_path)
        
        template_path = os.path.join(os.path.dirname(__file__), '../../templates/single_state/object_info_template.json')
        with open(template_path, 'r', encoding='utf-8') as f:
            self.templates = json.load(f)['object_info']
        
    def generate_questions(self) -> List[Dict[str, Any]]:
        questions = []
        
        for state_key in self.scene_data.keys():
            if state_key in ['room_static', 'object_mapping']:
                continue
                
            questions.extend(self._generate_object_distance_questions(state_key))
            questions.extend(self._generate_object_shape_questions(state_key))
            questions.extend(self._generate_object_color_questions(state_key))
            questions.extend(self._generate_object_height_questions(state_key))
            questions.extend(self._generate_object_room_questions(state_key)) # ?
            questions.extend(self._generate_object_num_questions(state_key))
            questions.extend(self._generate_object_receptacle_questions(state_key))
            questions.extend(self._generate_receptacle_content_questions(state_key))
        
        return questions
    

    def _check_obj_receptacle(self, obj1: Dict, obj2: Dict):
        if not obj1['receptacle'] and not obj2['receptacle']:
            return False

        if obj1['receptacle']:
            receptacleObjectIds =  obj1['receptacleObjectIds']
            if receptacleObjectIds is None:
                return False
            else:
                obj2_id = obj2['objectId']
                if obj2_id in receptacleObjectIds:
                    return obj1, obj2
                else:
                    return False

        if obj2['receptacle']:
            receptacleObjectIds =  obj2['receptacleObjectIds']
            if receptacleObjectIds is None:
                return False
            else:
                obj1_id = obj1['objectId']
                if obj1_id in receptacleObjectIds:
                    return obj2, obj1
                else:
                    return False
    
    def _generate_distance_estimation_questions(self, state_key: str, selected_pairs: List, all_distances: Dict, all_obj_pairs: Dict) -> List[Dict[str, Any]]:
        available_keys = ['type', 'type_in_room', 'in_receptacle', 'with_contents', 'size', 'position', 'attribute']
        question_template = self.templates['object_distance_estimation']['question_template']
        questions = []

        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)

        for pair in selected_pairs:
            obj1_names, obj2_names = all_obj_pairs[pair]
            obj1_name, obj1_key = self._select_name(obj1_names, available_keys, False) ##
            obj2_name, obj2_key = self._select_name(obj2_names, available_keys, False) ##
            if obj1_name is None or obj2_name is None:
                continue

            distance = all_distances[pair]

            if distance < 0.5:
                distance *= 100
                unit = 'cm'
            else:
                unit = 'm'
            
            num_choices = random.randint(4, 6)
            choices, correct_choice = self.create_range_choices(list(all_distances.values()), distance, unit, num_choices)
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
            clip1 = self.get_object_visible_frames(state_key, pair[0])
            clip2 = self.get_object_visible_frames(state_key, pair[1])

            clips = self.find_continous_clips(self.find_frame_intersections([clip1, clip2]))
            if not clips:
                continue
            clip = max(clips, key=len)
            start_frame = clip[0]
            end_frame = clip[-1]
            # clip_data = self.get_one_clip_data(clip, state_key, objs = [(pair[0], 0), (pair[1], 0)])
            # answer_source = [
            #     {'type': 'text', 'content': 'First, you need to identify the objects mentioned in the question. Then, look in the video for some recognizable objects whose standard dimensions you know. Use the known dimensions of these reference objects to estimate the distance between the specified objects.'},
            #     {'type': 'text', 'content': f'The video clip is extracted from frames {start_frame} to {end_frame} of the {video_num} video with a total length of {len_video} frames.'},
            #     {'type': 'video', 'content': clip_data, 'state': state_key},
            # ]

            related_frames = {state_key: self.find_frame_unions([clip1, clip2])}

            question_text = question_template.format(object1=obj1_name, object2=obj2_name, state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_choice,
                'question_type': self.templates['object_distance_estimation']['question_type'],
                'category': 'object_info',
                'subcategory': 'object_distance_estimation',
                'state': state_key,
                'capabilities': self.templates['object_distance_estimation']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })

        return questions

    def _generate_distance_shortest_comparison_questions(self, state_key: str, candidate_objs: List, clip: List, target_obj_name: str):
        question_template = self.templates['object_distance_shortest_comparison']['question_template']
        questions = []

        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)

        used_objects = set()
        while True:
            available_candidates = [obj for obj in candidate_objs if obj[1] not in used_objects]
            
            if len(available_candidates) < 4:
                break
            
            num_choices = min(random.randint(4, 6), len(available_candidates))
            selected_candidates = random.sample(available_candidates, num_choices)
            selected_candidates.sort(key=lambda x: x[2])
            closest_obj = selected_candidates[0]
            choices = [obj[1] for obj in selected_candidates]
            random.shuffle(choices)
            correct_answer = choices.index(closest_obj[1])
            
            answer_source = [] ###
            start_frame = clip[0]
            end_frame = clip[-1]
            # obj_ids = [(obj[0], 0) for obj in selected_candidates]
            # clip_data = self.get_one_clip_data(clip, state_key, objs = [(target_obj_id, 0)] + obj_ids)
            # start_frame = clip_data['frames'][0]['frame_id']
            # end_frame = clip_data['frames'][-1]['frame_id']
            # answer_source = [
            #     {'type': 'text', 'content': 'First, you need to identify the objects mentioned in the options.'},
            #     {'type': 'text', 'content': 'Then, look in the video for some recognizable objects whose standard dimensions you know, such as a standard door (typically around 2 meters in height) or a soda can. Use the known dimensions of these reference objects to estimate the distances between the specified object pairs. Finally, compare the distances of the object pairs listed in the options.'},
            #     {'type': 'text', 'content': f'The video clip is extracted from frames {start_frame} to {end_frame} of the {video_num} video with a total length of {len_video} frames.'},
            #     {'type': 'video', 'content': [clip_data], 'state': state_key},
            # ]

            related_frames = {state_key: set()}
            for frame_id in range(start_frame, end_frame + 1):
                related_frames[state_key].add(frame_id)
            related_frames[state_key] = sorted(list(related_frames[state_key]))
            
            question_text = question_template.format(object=target_obj_name, state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['object_distance_shortest_comparison']['question_type'],
                'category': 'object_info',
                'subcategory': 'object_distance_shortest_comparison',
                'state': state_key,
                'capabilities': self.templates['object_distance_shortest_comparison']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })
            
            for obj in selected_candidates:
                used_objects.add(obj[1])

        return questions

    def _generate_distance_longest_comparison_questions(self, state_key: str, candidate_objs: List, clip: List, target_obj_name: str):
        question_template = self.templates['object_distance_longest_comparison']['question_template']
        questions = []

        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)

        used_objects = set()
        while True:
            available_candidates = [obj for obj in candidate_objs if obj[1] not in used_objects]
            
            if len(available_candidates) < 4:
                break
            
            num_choices = min(random.randint(4, 6), len(available_candidates))
            selected_candidates = random.sample(available_candidates, num_choices)
            selected_candidates.sort(key=lambda x: x[2])
            farthest_obj = selected_candidates[-1]
            choices = [obj[1] for obj in selected_candidates]
            random.shuffle(choices)
            correct_answer = choices.index(farthest_obj[1])

            answer_source = [] ###
            start_frame = clip[0]
            end_frame = clip[-1]
            # obj_ids = [(obj[0], 0) for obj in selected_candidates]
            # clip_data = self.get_one_clip_data(clip, state_key, objs = [(target_obj_id, 0)] + obj_ids)
            # start_frame = clip_data['frames'][0]['frame_id']
            # end_frame = clip_data['frames'][-1]['frame_id']
            # answer_source = [
            #     {'type': 'text', 'content': 'First, you need to identify the objects mentioned in the options.'},
            #     {'type': 'text', 'content': 'Then, look in the video for some recognizable objects whose standard dimensions you know, such as a standard door (typically around 2 meters in height) or a soda can. Use the known dimensions of these reference objects to estimate the distances between the specified object pairs. Finally, compare the distances of the object pairs listed in the options.'},
            #     {'type': 'text', 'content': f'The video clip is extracted from frames {start_frame} to {end_frame} of the {video_num} video with a total length of {len_video} frames.'},
            #     {'type': 'video', 'content': clip_data, 'state': state_key},
            # ]

            related_frames = {state_key: set()}
            for frame_id in range(start_frame, end_frame + 1):
                related_frames[state_key].add(frame_id)
            related_frames[state_key] = sorted(list(related_frames[state_key]))
            
            question_text = question_template.format(object=target_obj_name, state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['object_distance_longest_comparison']['question_type'],
                'category': 'object_info',
                'subcategory': 'object_distance_longest_comparison',
                'state': state_key,
                'capabilities': self.templates['object_distance_longest_comparison']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })
            
            for obj in selected_candidates:
                used_objects.add(obj[1])

        return questions

    def _generate_distance_comparison_questions(self, state_key: str, room: Dict, room_objs: List, num_objs: int) -> List[Dict[str, Any]]:
        available_keys = ['type', 'type_in_room', 'in_receptacle', 'with_contents', 'size', 'position', 'attribute']
        questions = []

        for target_idx, target_obj in enumerate(room_objs):
            target_obj_id = target_obj['objectId']
            if target_obj_id not in self.object_names[state_key]:
                continue
                
            target_obj_names = self.object_names[state_key][target_obj_id]
            target_obj_name, target_obj_key = self._select_name(target_obj_names, available_keys) 
            if not target_obj_name:
                continue
            
            candidate_objs = []
            for i, obj in enumerate(room_objs):
                if i == target_idx:
                    continue
                
                if not self._check_obj_receptacle(target_obj, obj):
                    obj_id = obj['objectId']
                    if obj_id in self.object_names[state_key]:
                        obj_names = self.object_names[state_key][obj_id]
                        obj_name = None
                        for key in available_keys:
                            if obj_names[key] is not None:
                                obj_name = obj_names[key]
                                break
                        if obj_name:
                            distance = self.calculate_distance(target_obj['position'], obj['position'])
                            candidate_objs.append((obj_id, obj_name, distance))
            
            if len(candidate_objs) < 2:
                continue
            
            candidate_objs.sort(key=lambda x: x[2])
            
            clip = self.select_proper_room_frames(state_key, room)
            if not clip or len(clip) < 10:
                continue

            if random.random() * num_objs > 1:
                continue
            questions.extend(self._generate_distance_shortest_comparison_questions(state_key, candidate_objs, clip, target_obj_name))

            if random.random() * num_objs > 1:
                continue
            questions.extend(self._generate_distance_longest_comparison_questions(state_key, candidate_objs, clip, target_obj_name))

        

        return questions

    def _generate_object_distance_questions(self, state_key: str) -> List[Dict[str, Any]]:
        questions = []

        room_objects = self.room_objects[state_key]

        rooms = self.scene_data['room_static']['room_static_details']
        num_objs = 0
        for room in rooms:
            if room['room_name'] not in room_objects:
                continue
            num_objs += len(room_objects[room['room_name']])
        
        for room in rooms:
            if room['room_name'] not in room_objects:
                continue
            room_objs = room_objects[room['room_name']]
            if len(room_objs) < 2: 
                continue
                
            all_distances = {}
            all_obj_pairs = {}
            for i in range(len(room_objs)):
                for j in range(i+1, len(room_objs)):
                    check_res = self._check_obj_receptacle(room_objs[i], room_objs[j])
                    if not check_res:
                        obj1_id = room_objs[i]['objectId']
                        obj2_id = room_objs[j]['objectId']
                        obj1_names = self.object_names[state_key][obj1_id]
                        obj2_names = self.object_names[state_key][obj2_id]
                        d = self.calculate_distance(room_objs[i]['position'], room_objs[j]['position'])
                        if d > 0.1 and d < 5:
                            all_distances[(obj1_id, obj2_id)] = d
                            all_obj_pairs[(obj1_id, obj2_id)] = (obj1_names, obj2_names)

            selected_pairs = random.sample(list(all_distances.keys()), min(len(all_distances.keys()), max(1, int(10//len(rooms)))))
            questions.extend(self._generate_distance_estimation_questions(state_key, selected_pairs, all_distances, all_obj_pairs))
            
            if len(room_objs) < 4:
                continue
            questions.extend(self._generate_distance_comparison_questions(state_key, room, room_objs, num_objs))

        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    

    def _generate_object_shape_questions(self, state_key: str) -> List[Dict[str, Any]]:
        available_keys = ['type', 'type_in_room', 'in_receptacle', 'with_contents', 'size', 'position']
        question_template = self.templates['object_shape']['question_template']
        questions = []
        
        room_objects = self.room_objects[state_key]
        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)

        objects = []
        for objects_in_room in room_objects.values():
            for obj in objects_in_room:
                if "attributes" not in obj or obj['attributes']['shape'] == 'unknown':
                    continue
                
                obj_id = obj['objectId']
                if obj_id not in self.object_names[state_key]:
                    continue
                    
                obj_names = self.object_names[state_key][obj_id]
                obj_name, obj_key = self._select_name(obj_names, available_keys) ###
                if not obj_name:
                    continue
                    
                objects.append((obj_id, obj_name, obj["attributes"]["shape"]))

        selected_objects = random.sample(objects, min(5, len(objects)))

        for obj_id, obj_name, obj_shape in selected_objects:
            
            choices = [obj_shape]
            num_choices = random.randint(4, 6)
            
            random.shuffle(self.shape_options)
            for option in self.shape_options:
                if option != obj_shape and len(choices) < num_choices:
                    choices.append(option)

            original_correct_answer = obj_shape
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

            answer_source = [] ###    
            obj_visible_frames = self.get_object_visible_frames(state_key, obj_id)
            clips = self.find_continous_clips(obj_visible_frames)
            if not clips:
                continue
            clip = max(clips, key=len)
            if len(clip) < 10:
                continue
            start_frame = clip[0]
            end_frame = clip[-1]
            # clip_data = self.get_one_clip_data(clip, state_key, objs = [(obj_id, 0)])
            
            # answer_source = [
            #     {'type': 'text', 'content': 'First, you need to identify the object mentioned in the question.'},
            #     {'type': 'text', 'content': 'After locating the object, compare its shape with those listed in the options and select the one that is the closest match.'},
            #     {'type': 'video', 'content': clip_data, 'state': state_key},
            #     {'type': 'text', 'content': f'The video clip is extracted from frames {start_frame} to {end_frame} of the {video_num} video with a total length of {len_video} frames.'},
            # ]

            related_frames = {state_key: set()}
            for frame_id in range(start_frame, end_frame + 1):
                related_frames[state_key].add(frame_id)
            related_frames[state_key] = sorted(list(related_frames[state_key]))
            
            question_text = question_template.format(object_id=obj_name, state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['object_shape']['question_type'],
                'category': 'object_info',
                'subcategory': 'object_shape',
                'state': state_key,
                'capabilities': self.templates['object_shape']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    

    def _generate_object_color_questions(self, state_key: str) -> List[Dict[str, Any]]:
        available_keys = ['type', 'type_in_room', 'in_receptacle', 'with_contents', 'size', 'position']
        question_template = self.templates['object_color']['question_template']
        questions = []

        room_objects = self.room_objects[state_key]
        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)

        objects = []
        for objects_in_room in room_objects.values():
            for obj in objects_in_room:
                if "attributes" not in obj or obj['attributes']['color'] == 'unknown':
                    continue
                
                obj_id = obj['objectId']
                if obj_id not in self.object_names[state_key]:
                    continue
                    
                obj_names = self.object_names[state_key][obj_id]
                obj_name, obj_key = self._select_name(obj_names, available_keys) ##    
                if not obj_name:
                    continue
                    
                objects.append((obj_id, obj, obj_name))

        selected_objects = random.sample(objects, min(5, len(objects)))

        for obj_id, obj, obj_name in selected_objects:
            correct_color = obj["attributes"]["color"]
            choices = [correct_color]
            num_choices = random.randint(4, 6)

            random.shuffle(self.color_options)
            for option in self.color_options:
                if option != correct_color and len(choices) < num_choices:
                    choices.append(option)

            original_correct_answer = correct_color
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

            answer_source = [] ###
            obj_visible_frames = self.get_object_visible_frames(state_key, obj_id)
            clips = self.find_continous_clips(obj_visible_frames)
            if not clips:
                continue
            clip = max(clips, key=len)
            if len(clip) < 10:
                continue
            start_frame = clip[0]
            end_frame = clip[-1]
            # clip_data = self.get_one_clip_data(clip, state_key, objs = [(obj_id, 0)])

            # answer_source = [
            #     {'type': 'text', 'content': 'First, you need to identify the object mentioned in the question.'},
            #     {'type': 'text', 'content': 'After locating the object, compare its color with those listed in the options and select the one that is the closest match.'},
            #     {'type': 'video', 'content': clip_data, 'state': state_key},
            #     {'type': 'text', 'content': f'The video clip is extracted from frames {start_frame} to {end_frame} of the {video_num} video with a total length of {len_video} frames.'},
            # ]

            related_frames = {state_key: set()}
            for frame_id in range(start_frame, end_frame + 1):
                related_frames[state_key].add(frame_id)
            related_frames[state_key] = sorted(list(related_frames[state_key]))

            question_text = question_template.format(object_id=obj_name, state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['object_color']['question_type'],
                'category': 'object_info',
                'subcategory': 'object_color',
                'state': state_key,
                'capabilities': self.templates['object_color']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })

        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    

    def _generate_object_height_questions(self, state_key: str) -> List[Dict[str, Any]]:
        available_keys = ['type', 'type_in_room', 'in_receptacle', 'with_contents', 'size', 'position', 'attribute']

        question_template = self.templates['object_height']['question_template']
        questions = []
        
        room_objects = self.room_objects[state_key]
        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)

        objects = []
        all_heights_m = []
        all_heights_cm = []
        for objects_in_room in room_objects.values():
            for obj in objects_in_room:
                if not obj.get('moveable', False):
                    continue
                    
                if 'axisAlignedBoundingBox' in obj and 'y' in obj['axisAlignedBoundingBox'].get('size', {}):
                    h = obj['axisAlignedBoundingBox']['size']['y']
                    all_heights_m.append(h)
                    all_heights_cm.append(h * 100)
                
                obj_id = obj['objectId']
                if obj_id in self.object_names[state_key]:
                    obj_names = self.object_names[state_key][obj_id]
                    obj_name, obj_key = self._select_name(obj_names, available_keys) ##
                    if obj_name:
                        objects.append((obj_id, obj, obj_name))
        
        selected_objects = random.sample(objects, min(5, len(objects)))
        for obj_id, obj, obj_name in selected_objects:
            bbox = obj['axisAlignedBoundingBox']
            size = bbox.get('size', {})
            
            height = size['y']
            
            if height < 0.1:
                continue

            if height < 0.5:
                unit = "cm"
                all_heights = all_heights_cm
                height = height * 100
            else:
                unit = 'm'
                all_heights = all_heights_m

            num_choices = random.randint(4, 6)
            choices, original_correct_answer = self.create_range_choices(all_heights, height, unit, num_choices)
            if not choices:
                continue

            if random.random() < 1/3:
                is_hallucination = True
                correct_answer_value = self.no_valid_option
                choices.pop(original_correct_answer)
                choices.append(correct_answer_value)
            else:
                is_hallucination = False
                correct_answer_value = copy.copy(choices[original_correct_answer])
                candidate_remove_ids = [i for i, c in enumerate(choices) if i != original_correct_answer]
                if candidate_remove_ids:
                    remove_id = random.choice(candidate_remove_ids)
                    choices.pop(remove_id)
                choices.append(self.no_valid_option)

            if correct_answer_value not in choices:
                import pdb;pdb.set_trace()
            random.shuffle(choices)
            correct_answer = choices.index(correct_answer_value)

            answer_source = [] ###
            obj_visible_frames = self.get_object_visible_frames(state_key, obj_id)
            clips = self.find_continous_clips(obj_visible_frames)
            if not clips:
                continue
            clip = max(clips, key=len)
            if len(clip) < 10:
                continue
            start_frame = clip[0]
            end_frame = clip[-1]
            # clip_data = self.get_one_clip_data(clip, state_key, objs = [(obj_id, 0)])
        
            # answer_source = [
            #     {'type': 'text', 'content': 'First, you need to identify the object mentioned in the question.'},
            #     {'type': 'text', 'content': 'Then, estimate the object\'s height based on common knowledge and its surrounding environment. Finally, compare the estimated height with the options and select the one that is the closest match.'},
            #     {'type': 'video', 'content': clip_data, 'state': state_key},
            #     {'type': 'text', 'content': f'The video clip is extracted from frames {start_frame} to {end_frame} of the {video_num} video with a total length of {len_video} frames.'},
            # ]
            
            related_frames = {state_key: set()}
            for frame_id in range(start_frame, end_frame + 1):
                related_frames[state_key].add(frame_id)
            related_frames[state_key] = sorted(list(related_frames[state_key]))
            
            question_text = question_template.format(object_id=obj_name, state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['object_height']['question_type'],
                'category': 'object_info',
                'subcategory': 'object_height',
                'state': state_key,
                'capabilities': self.templates['object_height']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions


    def _generate_object_room_questions(self, state_key: str) -> List[Dict[str, Any]]:
        available_room_keys = ['type', 'biggest_room', 'smallest_room', 'biggest_type', 'smallest_type', 'shape']
        available_object_keys = ['type', 'in_receptacle', 'with_contents', 'size', 'position', 'attribute']
        question_template = self.templates['object_room_location']['question_template']
        questions = []
        
        room_objects = self.room_objects[state_key]
        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)

        rooms = self.scene_data['room_static']['room_static_details']
        available_rooms = {}
        for room in rooms:
            room_id = room['room_name']
            if room_id not in self.room_names[state_key]:
                continue
                
            room_names = self.room_names[state_key][room_id]
            room_name, room_key = self._select_name(room_names, available_room_keys)
            if not room_name:
                continue
            
            available_rooms[room_name] = (room_objects.get(room['room_name'], []), room)

        if len(available_rooms) < 4:
            return questions
        
        num_objects = 0
        for room_name, room_objs in available_rooms.items():
            num_objects += len(room_objs[0])
        
        for room_name, (room_objs, room) in available_rooms.items():
            clip = self.select_proper_room_frames(state_key, room)
            if not clip or len(clip) < 10:
                continue
            for obj in room_objs:
                if random.random() * num_objects > 5:
                    continue
                obj_id = obj['objectId']
                if obj_id not in self.object_names[state_key]:
                    continue
                    
                obj_names = self.object_names[state_key][obj_id]
                obj_name, obj_key = self._select_name(obj_names, available_object_keys) ##
                if not obj_name:
                    continue
                
                correct_room = room_name
                choices = [correct_room]
                num_choices = min(random.randint(4, 6), len(available_rooms))
                for room_name in available_rooms.keys():
                    if room_name not in choices and len(choices) < num_choices:
                        choices.append(room_name)

                original_correct_answer = correct_room
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
                
                answer_source = [] ###
                start_frame = clip[0]
                end_frame = clip[-1]
                # clip_data = self.get_one_clip_data(clip, state_key, objs = [(obj_id, 0)])

                # answer_source = [
                #     {'type': 'text', 'content': 'First, you need to identify the object mentioned in the question.'},
                #     {'type': 'text', 'content': 'Then determine which of the rooms listed in the options is closest to the room where the object is located.'},
                #     {'type': 'video', 'content': clip_data, 'state': state_key},
                #     {'type': 'text', 'content': f'The video clip is extracted from frames {start_frame} to {end_frame} of the {video_num} video with a total length of {len_video} frames.'},
                # ]

                related_frames = {state_key: set()}
                for frame_id in range(start_frame, end_frame + 1):
                    related_frames[state_key].add(frame_id)
                related_frames[state_key] = sorted(list(related_frames[state_key]))

                question_text = question_template.format(object_id=obj_name, state=state_key)
                questions.append({
                    'question': question_text,
                    'choices': choices,
                    'correct_answer': correct_answer,
                    'question_type': self.templates['object_room_location']['question_type'],
                    'category': 'object_info',
                    'subcategory': 'object_room_location',
                    'state': state_key,
                    'capabilities': self.templates['object_room_location']['capabilities'],
                    'answer_source': answer_source,
                    'related_frames': related_frames,
                    'hallucination': is_hallucination
                })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions


    def _generate_room_object_num_questions(self, state_key: str, room_object_type_distribute: Dict, name_room_dict: Dict) -> List[Dict[str, Any]]:
        question_template = self.templates['object_type_num_in_room']['question_template']
        questions = []

        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)

        all_num = []
        for room_name, object_type_distribute in room_object_type_distribute.items():
            for object_type, obj_ids in object_type_distribute.items():
                all_num.append(len(obj_ids))
        all_num = set(all_num)
        if not all_num:
            all_num = {0}

        num_types = 0
        simplified_room_object_type_distribute = {}
        for room_name, object_type_distribute in room_object_type_distribute.items():
            num_to_object_types = {}
            for object_type, obj_ids in object_type_distribute.items():
                if len(obj_ids) not in num_to_object_types:
                    num_to_object_types[len(obj_ids)] = []
                num_to_object_types[len(obj_ids)].append(object_type)
            
            simplified_object_type_distribute = {}
            for num, object_types in num_to_object_types.items():
                selected_object_type = random.choice(object_types)
                simplified_object_type_distribute[selected_object_type] = object_type_distribute[selected_object_type]
            num_types += sum(len(obj_ids) for obj_ids in simplified_object_type_distribute.values())
            simplified_room_object_type_distribute[room_name] = simplified_object_type_distribute


        for room_name, object_type_distribute in simplified_room_object_type_distribute.items():
            if random.random() * num_types > 5:
                continue
            clip = self.select_proper_room_frames(state_key, name_room_dict[room_name])
            if not clip or len(clip) < 10:
                continue
            for object_type, obj_ids in object_type_distribute.items():
                if random.random() * num_types > 3:
                    continue
                correct_num = len(obj_ids)
                max_val = max(all_num)
                min_val = min(all_num)

                num_choices = random.randint(3, 5)
                candidate_pool = list(range(min_val, max_val + 1))
                
                if len(candidate_pool) < num_choices:
                    # Add numbers below min_val (but not below 0)
                    extend_below = max(0, min_val - (num_choices - len(candidate_pool)))
                    candidate_pool.extend(range(extend_below, min_val))
                    
                    # Add numbers above max_val if still needed
                    if len(candidate_pool) < num_choices:
                        extend_above = max_val + 1 + (num_choices - len(candidate_pool))
                        candidate_pool.extend(range(max_val + 1, extend_above))
                
                if correct_num in candidate_pool:
                    candidate_pool.remove(correct_num)
                
                selected_choices = random.sample(candidate_pool, min(num_choices - 1, len(candidate_pool)))
                choices = [correct_num] + selected_choices

                is_hallucination = random.random() < 1/3
                if is_hallucination:
                    choices.remove(correct_num)
                    choices.append(self.no_valid_option)
                    correct_num = self.no_valid_option
                else:
                    candidate_remove_ids = [i for i, choice in enumerate(choices) if choice != correct_num]
                    if candidate_remove_ids: # Ensure there are other choices to remove
                        remove_id = random.choice(candidate_remove_ids)
                        choices.remove(choices[remove_id])
                    choices.append(self.no_valid_option)

                random.shuffle(choices)
                correct_answer = choices.index(correct_num)
                
                answer_source = [] ###
                start_frame = clip[0]
                end_frame = clip[-1]
                # objs = [(obj, 0) for obj in obj_ids]
                # clip_data = self.get_one_clip_data(clip, state_key, objs)

                # answer_source = [
                #     {'type': 'text', 'content': 'First, you need to identify the room mentioned in the question.'},
                #     {'type': 'text', 'content': 'Then count the number of objects belonging to the specified category within that room.'},
                #     {'type': 'video', 'content': clip_data, 'state': state_key},
                #     {'type': 'text', 'content': f'The video clip is extracted from frames {start_frame} to {end_frame} of the {video_num} video with a total length of {len_video} frames.'},
                # ]

                related_frames = {state_key: set()}
                for frame_id in range(start_frame, end_frame + 1):
                    related_frames[state_key].add(frame_id)
                related_frames[state_key] = sorted(list(related_frames[state_key]))

                question_text = question_template.format(object_type=object_type, room_name=room_name, state=state_key)
                questions.append({
                    'question': question_text,
                    'choices': choices,
                    'correct_answer': correct_answer,
                    'question_type': self.templates['object_type_num_in_room']['question_type'],
                    'category': 'object_info',
                    'subcategory': 'object_type_num_in_room',
                    'state': state_key,
                    'capabilities': self.templates['object_type_num_in_room']['capabilities'],
                    'answer_source': answer_source,
                    'related_frames': related_frames,
                    'hallucination': is_hallucination
                })

        return questions

    def _generate_scene_object_num_questions(self, state_key: str, room_object_type_distribute: Dict) -> List[Dict[str, Any]]:
        question_template = self.templates['object_type_num_in_scene']['question_template']
        questions = []

        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)

        clip = list(range(len(self.scene_data[state_key]['agent_trajectory'])))

        all_object_type_distribute = {}
        for _, object_type_distribute in room_object_type_distribute.items():
            for object_type, obj_ids in object_type_distribute.items():
                if object_type in all_object_type_distribute.keys():
                    all_object_type_distribute[object_type].extend(obj_ids)
                else:
                    all_object_type_distribute[object_type] = obj_ids.copy()

        all_num = []
        for object_type, obj_ids in all_object_type_distribute.items():
            all_num.append(len(obj_ids))
        all_num = set(all_num)

        simplified_all_object_type_distribute = {}
        num_to_object_types = {}
        for object_type, obj_ids in all_object_type_distribute.items():
            num = len(obj_ids)
            if num not in num_to_object_types:
                num_to_object_types[num] = []
            num_to_object_types[num].append(object_type)
        
        for num, object_types in num_to_object_types.items():
            selected_object_type = random.choice(object_types)
            simplified_all_object_type_distribute[selected_object_type] = all_object_type_distribute[selected_object_type]

        for object_type, obj_ids in simplified_all_object_type_distribute.items():
            if random.random() * len(simplified_all_object_type_distribute) > 5:
                continue
            correct_num = len(obj_ids)
            choices = [correct_num]
            max_val = max(all_num)
            min_val = min(all_num)

            # Add random numbers within the range [min_val, max_val] to choices
            num_choices = random.randint(3, 5)
            available_range = max_val - min_val + 1
            
            # If not enough values in range, expand the range
            if num_choices > available_range:
                # Calculate how much to expand
                expand_needed = num_choices - available_range
                # Expand both sides equally, ensuring min_val doesn't go below 0
                expand_each = expand_needed // 2 + (1 if expand_needed % 2 else 0)
                min_val = max(0, min_val - expand_each)
                max_val = max_val + (expand_needed - (min_val - max(0, min_val - expand_each)))
            
            # Create a pool of candidate numbers
            candidate_pool = list(range(min_val, max_val + 1))
            
            # Remove the correct answer from pool to avoid duplication
            if correct_num in candidate_pool:
                candidate_pool.remove(correct_num)
            
            # If we need more numbers, expand the pool
            if len(candidate_pool) < num_choices - 1:
                # Add numbers below min_val (but not below 0)
                extend_below = max(0, min_val - (num_choices - 1 - len(candidate_pool)))
                candidate_pool.extend(range(extend_below, min_val))
                
                # Add numbers above max_val if still needed
                if len(candidate_pool) < num_choices - 1:
                    extend_above = max_val + 1 + (num_choices - 1 - len(candidate_pool))
                    candidate_pool.extend(range(max_val + 1, extend_above))
            
            # Randomly select choices from the pool
            selected_choices = random.sample(candidate_pool, min(num_choices - 1, len(candidate_pool)))
            choices.extend(selected_choices)

            is_hallucination = random.random() < 1/3
            if is_hallucination:
                choices.remove(correct_num)
                choices.append(self.no_valid_option)
                correct_num = self.no_valid_option
            else:
                candidate_remove_ids = [i for i, choice in enumerate(choices) if choice != correct_num]
                if candidate_remove_ids: # Ensure there are other choices to remove
                    remove_id = random.choice(candidate_remove_ids)
                    choices.remove(choices[remove_id])
                choices.append(self.no_valid_option)

            random.shuffle(choices)
            correct_answer = choices.index(correct_num)

            answer_source = [] ###
            start_frame = clip[0]
            end_frame = clip[-1]
            # objs = [(obj, 0) for obj in obj_ids]
            # clip_data = self.get_one_clip_data(clip, state_key, objs)
            
            # answer_source = [
            #     {'type': 'text', 'content': 'You need to count the number of objects belonging to the specified category within the scene.'},
            #     {'type': 'video', 'content': clip_data, 'state': state_key}
            # ]

            related_frames = {state_key: set()}
            for frame_id in range(start_frame, end_frame + 1):
                related_frames[state_key].add(frame_id)
            related_frames[state_key] = sorted(list(related_frames[state_key]))
            
            question_text = question_template.format(object_type=object_type, state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['object_type_num_in_scene']['question_type'],
                'category': 'object_info',
                'subcategory': 'object_type_num_in_scene',
                'state': state_key,
                'capabilities': self.templates['object_type_num_in_scene']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })


        return questions

    def _generate_object_num_questions(self, state_key: str) -> List[Dict[str, Any]]:
        vailable_room_keys = ['type', 'biggest_room', 'smallest_room', 'biggest_type', 'smallest_type', 'shape']
        questions = []

        room_objects = self.room_objects[state_key]

        rooms = self.scene_data['room_static']['room_static_details']
        room_object_type_distribute = {}
        name_room_dict = {}
        all_room_object_types = set()
        for room in rooms:
            room_id = room['room_name']
            if room_id not in self.room_names[state_key]:
                continue

            room_names = self.room_names[state_key][room_id]
            room_name, room_key = self._select_name(room_names, vailable_room_keys)
            if not room_name:
                continue
            
            name_room_dict[room_name] = room
            objects_in_room = room_objects.get(room_id, [])
            object_type_distribute = {}
            for obj in objects_in_room:
                object_type = obj['objectType']
                all_room_object_types.add(object_type)
                if object_type in object_type_distribute.keys():
                    # object_type_distribute[object_type] += 1
                    object_type_distribute[object_type].append(obj['objectId'])
                else:
                    # object_type_distribute[object_type] = 1
                    object_type_distribute[object_type] = [obj['objectId']]
                
            room_object_type_distribute[room_name] = object_type_distribute

        questions.extend(self._generate_room_object_num_questions(state_key, room_object_type_distribute, name_room_dict))

        questions.extend(self._generate_scene_object_num_questions(state_key, room_object_type_distribute))
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions


    def _generate_object_receptacle_questions(self, state_key: str) -> List[Dict[str, Any]]:
        object_available_keys = ['type', 'with_contents', 'size', 'position', 'attribute']
        receptacle_available_keys = ['type', 'type_in_room', 'in_receptacle']
        question_template = self.templates['object_receptacle_location']['question_template']
        questions = []
        
        room_objects = self.room_objects[state_key]

        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = self.get_video_num(state_key)

        all_receptacles = []
        available_pairs = []
        for _, objects in room_objects.items():
            available_objects = []
            for idx, obj in enumerate(objects):
                obj_id = obj['objectId']
                if obj_id not in self.object_names[state_key]:
                    continue
                    
                obj_names = self.object_names[state_key][obj_id]
                if obj['receptacle']:
                    keys_to_use = receptacle_available_keys
                else:
                    keys_to_use = object_available_keys
                obj_name, obj_key = self._select_name(obj_names, keys_to_use) ##
                if obj_name:
                    available_objects.append(idx)
                    if obj['receptacle']:
                        all_receptacles.append(obj_name)
            
            for i in range(len(available_objects)):
                for j in range(i+1, len(available_objects)):
                    obj1 = objects[available_objects[i]]
                    obj2 = objects[available_objects[j]]
                    res = self._check_obj_receptacle(obj1, obj2)
                    if res:
                        available_pairs.append(res)
        if len(all_receptacles) < 3:
            return questions

        for receptacle, object in available_pairs:
            if random.random() * len(available_pairs) > 5:
                continue
            receptacle_id = receptacle['objectId']
            object_id = object['objectId']
            
            if receptacle_id not in self.object_names[state_key] or object_id not in self.object_names[state_key]:
                continue
                
            receptacle_names = self.object_names[state_key][receptacle_id]
            object_names = self.object_names[state_key][object_id]
            receptacle_name, receptacle_key = self._select_name(receptacle_names, receptacle_available_keys) ##
            object_name, object_key = self._select_name(object_names, object_available_keys) ##
            if not receptacle_name or not object_name:
                continue

            choices = [receptacle_name]
            num_choices = min(len(all_receptacles), random.randint(3, 5))
            random.shuffle(all_receptacles)
            for candidate_receptacle in all_receptacles:
                if candidate_receptacle not in choices:
                    choices.append(candidate_receptacle)
                if len(choices) >= num_choices:
                    break
            
            is_hallucination = random.random() < 1/3
            if is_hallucination:
                choices.remove(receptacle_name)
                choices.append(self.no_valid_option)
                correct_choices = self.no_valid_option
            else:
                candidate_remove_ids = [i for i, choice in enumerate(choices) if choice != receptacle_name]
                remove_id = random.choice(candidate_remove_ids)
                choices.remove(choices[remove_id])
                choices.append(self.no_valid_option)
                correct_choices = receptacle_name
            random.shuffle(choices)
            correct_answer = choices.index(correct_choices)

            answer_source = [] ###
            frames = self.get_object_visible_frames(state_key, object_id)
            clips = self.find_continous_clips(frames)
            if not clips:
                continue
            clip = max(clips, key = len)
            start_frame = clip[0]
            end_frame = clip[-1]
            # clip_data = self.get_one_clip_data(clip, state_key, [(object_id, 0), (receptacle_id, 0)])
            
            # answer_source = [
            #     {'type': 'text', 'content': 'First, you need to identify the specific object mentioned in the question.'},
            #     {'type': 'text', 'content': 'Then answer the question based on the objects placed above or inside it.'},
            #     {'type': 'video', 'content': clip_data, 'state': state_key},
            #     {'type': 'text', 'content': f'The video clip is extracted from frames {start_frame} to {end_frame} of the {video_num} video with a total length of {len_video} frames.'},
            # ]

            related_frames = {state_key: set()}
            for frame_id in range(start_frame, end_frame + 1):
                related_frames[state_key].add(frame_id)
            related_frames[state_key] = sorted(list(related_frames[state_key]))

            question_text = question_template.format(object_id=object_name, state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['object_receptacle_location']['question_type'],
                'category': 'object_info',
                'subcategory': 'object_receptacle_location',
                'state': state_key,
                'capabilities': self.templates['object_receptacle_location']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': False
            })
        
        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
    
    
    def _generate_receptacle_content_questions(self, state_key: str) -> List[Dict[str, Any]]:
        object_available_keys = ['type', 'with_contents', 'size', 'position', 'attribute']
        receptacle_available_keys = ['type', 'type_in_room', 'in_receptacle']
        question_template = self.templates['receptacle_contents']['question_template']
        questions = []
        
        availabel_receptacles = []
        available_objects = {}
        room_objects = self.room_objects[state_key]
        
        trajectory = self.scene_data[state_key]['agent_trajectory']
        len_video = len(trajectory)
        video_num = 'first' if state_key.split('_')[-1] == '0' else 'second'

        for _, objects in room_objects.items():
            for obj in objects:
                obj_id = obj['objectId']
                if obj_id not in self.object_names[state_key]:
                    continue
                    
                obj_names = self.object_names[state_key][obj_id]                
                if obj['receptacle']:
                    keys_to_use = receptacle_available_keys
                else:
                    keys_to_use = object_available_keys
                obj_name, obj_key = self._select_name(obj_names, keys_to_use) ##
                if not obj_name:
                    continue
                    
                if obj['receptacle']:
                    availabel_receptacles.append((obj_name, obj))
                else:
                    available_objects[obj['objectId']] = (obj_name, obj)

        available_obj_ids = list(available_objects.keys())

        for receptacle_name, receptacle in availabel_receptacles:
            if random.random() * len(availabel_receptacles) > 5:
                continue
            all_correct_choices = []
            all_correct_obj_ids = []
            objects = receptacle['receptacleObjectIds']
            for obj in objects:
                if obj in available_obj_ids:
                    all_correct_choices.append(available_objects[obj][0])
                    all_correct_obj_ids.append(obj)
            if len(all_correct_choices) == 0:
                continue
            
            correct_idx = random.sample(list(range(len(all_correct_choices))), min(len(all_correct_choices), 5))
            correct_choices = [all_correct_choices[i] for i in correct_idx]
            correct_obj_ids = [all_correct_obj_ids[i] for i in correct_idx]
            choices = correct_choices.copy()
            num_choices = max(len(choices), random.randint(3, 5))
            
            random.shuffle(available_obj_ids)
            for obj_id in available_obj_ids:
                if available_objects[obj_id][0] not in correct_choices and len(choices) < num_choices:
                    choices.append(available_objects[obj_id][0])
            
            if len(correct_choices) <= 2 and random.random() < 1/3:
                is_hallucination = True
                for choice in correct_choices:
                    choices.remove(choice)
                choices.append(self.no_valid_option)
                correct_choices = [self.no_valid_option]
            else:
                is_hallucination = False
                candidate_remove_ids = [i for i, choice in enumerate(choices) if choice not in correct_choices]
                if candidate_remove_ids:
                    remove_id = random.choice(candidate_remove_ids)
                    choices.remove(choices[remove_id])
                choices.append(self.no_valid_option)
            
            random.shuffle(choices)
            correct_answer = [i for i, choice in enumerate(choices) if choice in correct_choices]

            answer_source = [] ###
            all_frames = []
            for obj_id in correct_obj_ids:
                frames = self.get_object_visible_frames(state_key, obj_id)
                all_frames.append(frames)
            inter_obj_frames = self.find_frame_intersections(all_frames)
            if inter_obj_frames:
                clips = self.find_continous_clips(inter_obj_frames)
            else:
                union_obj_frames = self.find_frame_unions(all_frames)
                clips = self.find_continous_clips(union_obj_frames)
            if not clips:
                continue
            clip = max(clips, key = len)
            start_frame = clip[0]
            end_frame = clip[-1]

            # objs = [(receptacle['objectId'], 0)]
            # for obj_id in correct_obj_ids:
            #     objs.append((obj_id, 0))
            # clip_data = self.get_one_clip_data(clip, state_key, objs)

            # answer_source = [
            #     {'type': 'text', 'content': 'First, you need to identify the specific object mentioned in the question.'},
            #     {'type': 'text', 'content': 'Then answer the question based on the objects placed above or inside it.'},
            #     {'type': 'video', 'content': clip_data, 'state': state_key},
            #     {'type': 'text', 'content': f'The video clip is extracted from frames {start_frame} to {end_frame} of the {video_num} video with a total length of {len_video} frames.'}
            # ]

            related_frames = {state_key: set()}
            for frame_id in range(start_frame, end_frame + 1):
                related_frames[state_key].add(frame_id)
            related_frames[state_key] = sorted(list(related_frames[state_key]))

            question_text = question_template.format(receptacle_id=receptacle_name, state=state_key)
            questions.append({
                'question': question_text,
                'choices': choices,
                'correct_answer': correct_answer,
                'question_type': self.templates['receptacle_contents']['question_type'],
                'category': 'object_info',
                'subcategory': 'receptacle_contents',
                'state': state_key,
                'capabilities': self.templates['receptacle_contents']['capabilities'],
                'answer_source': answer_source,
                'related_frames': related_frames,
                'hallucination': is_hallucination
            })

        print(inspect.currentframe().f_code.co_name, len(questions))
        return questions
