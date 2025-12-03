#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "output"

SINGLE_STATE_TYPES = {
    'scene_info': {
        'name': '场景信息',
        'description': '关于房间面积、类型、功能等的问题',
        'template_file': TEMPLATES_DIR / "single_state" / "scene_info_template.json",
        'generator_class': 'SceneInfoGenerator'
    },
    'object_info': {
        'name': '对象信息',
        'description': '关于对象属性、位置、关系等的问题',
        'template_file': TEMPLATES_DIR / "single_state" / "object_info_template.json",
        'generator_class': 'ObjectInfoGenerator'
    },
    'agent_explore': {
        'name': '代理探索',
        'description': '关于代理移动路径、转向等的问题',
        'template_file': TEMPLATES_DIR / "single_state" / "agent_explore_template.json",
        'generator_class': 'AgentExploreGenerator'
    },
}

MULTI_STATE_TYPES = {
    'object_changes': {
        'name': '对象变化',
        'description': '关于对象变化的问题',
        'template_file': TEMPLATES_DIR / "multi_state" / "object_changes_template.json",
        'generator_class': 'ObjectChangesGenerator'
    },
    'multi_states_agent_explore': {
        'name': '代理探索',
        'description': '关于代理在多状态间的探索行为',
        'template_file': TEMPLATES_DIR / "multi_state" / "agent_explore_template.json",
        'generator_class': 'AgentExploreGenerator'
    },
}
