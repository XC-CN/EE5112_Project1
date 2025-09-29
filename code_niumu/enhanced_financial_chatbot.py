#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版基于规则的金融聊天机器人系统
支持配置文件、更好的实体识别和上下文管理
"""

import re
import json
import random
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class EnhancedFinancialChatbot:
    """增强版基于规则的金融聊天机器人"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", model_backend: str = "hf"):
        """
        初始化聊天机器人
        
        Args:
            model_name: 预训练模型名称，支持以下强大模型：
                - "THUDM/chatglm2-6b" (推荐): ChatGLM2-6B，强大的中英双语对话模型
                - "THUDM/chatglm-6b": ChatGLM-6B，清华开源对话模型
                - "baichuan-inc/Baichuan2-7B-Chat": 百川2-7B对话模型
                - "Qwen/Qwen-7B-Chat": 阿里通义千问7B对话模型
                - "microsoft/DialoGPT-medium": 原始DialoGPT模型
                - "models/generative-finetune": 本地微调模型
        """
        self.model_name = model_name
        self.model_backend = model_backend  # "hf" 或 "reddit"
        self.config = self._get_default_config()
        self.financial_data = None
        self.conversation_history = []
        
        # 加载预训练模型（可选）
        self.use_pretrained = False
        if self.model_backend == "hf":
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                # 设置pad token以避免attention mask警告
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.model.eval()
                self.use_pretrained = True
                print(f"Successfully loaded HF pretrained model: {model_name}")
            except Exception as e:
                print(f"Failed to load HF pretrained model: {e}")
                self.use_pretrained = False
        elif self.model_backend == "reddit":
            # 初始化 Reddit 字符级模型
            try:
                self._init_reddit_model(save_dir=os.path.join(os.path.dirname(__file__), 'models', 'reddit'))
                self.use_pretrained = True
                print("Successfully loaded Reddit TF checkpoint model")
            except Exception as e:
                print(f"Failed to load Reddit TF checkpoint model: {e}")
                self.use_pretrained = False
    
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置 - 完全基于数据集构建"""
        return {
            "financial_rules": {
                "investment": {"patterns": [], "keywords": []},
                "loan": {"patterns": [], "keywords": []},
                "insurance": {"patterns": [], "keywords": []},
                "banking": {"patterns": [], "keywords": []},
                "currency": {"patterns": [], "keywords": []},
                "market": {"patterns": [], "keywords": []},
                "tax": {"patterns": [], "keywords": []},
                "regulation": {"patterns": [], "keywords": []}
            },
            "response_templates": {
                "investment": [],
                "loan": [],
                "insurance": [],
                "banking": [],
                "currency": [],
                "market": [],
                "tax": [],
                "regulation": [],
                "Default reply": ["Thank you for your inquiry. I am a financial assistant."]
            },
            "financial_keywords": {
                "investment_products": ["stocks", "funds", "bonds", "futures", "forex", "gold", "real estate", "wealth management"],
                "loan_types": ["mortgage", "auto loan", "credit loan", "business loan", "consumer loan", "secured loan"],
                "insurance_types": ["life insurance", "health insurance", "property insurance", "auto insurance", "accident insurance", "critical illness"],
                "banking_services": ["account opening", "transfer", "credit card", "online banking", "mobile banking", "ATM", "deposit"],
                "currencies": ["CNY", "USD", "EUR", "JPY", "GBP", "HKD"],
                "risk_levels": ["low risk", "medium risk", "high risk", "conservative", "balanced", "aggressive"],
                "time_horizon": ["short-term", "mid-term", "long-term", "1 year", "3 years", "5 years", "10 years"],
                "amount_range": ["small amount", "medium amount", "large amount", "ten thousand", "hundred thousand", "million"]
            },
            "default_values": {
                "suggestion": "Diversified investment portfolio including stocks, bonds and money market funds",
                "advice": "Please choose appropriate products based on your risk tolerance",
                "factors": "Risk, return, liquidity and other factors",
                "investment_type": "stocks",
                "product": "funds",
                "outlook": "relatively optimistic",
                "recommendation": "Invest cautiously and diversify risks",
                "loan_type": "personal loan",
                "rate": "4.5%",
                "conditions": "Good credit history and stable income source",
                "amount": "10",
                "term": "36",
                "monthly_payment": "3000",
                "documents": "ID card, income certificate, bank statements, etc.",
                "approval_rate": "relatively high",
                "insurance_type": "life insurance",
                "products": "Term life insurance, whole life insurance, etc.",
                "terms": "Insurance liability, exemption clauses, etc.",
                "process": "Report, submit materials, review, claim settlement",
                "premium": "5000",
                "time": "15",
                "fees": "Charged by amount ratio, online banking transfer is free",
                "currency": "USD",
                "trend": "relatively stable",
                "analysis": "The market overall performance is stable, some sectors are active",
                "indicators": "GDP growth rate, inflation rate, employment data",
                "industry": "finance",
                "impact": "Increased policy support, good development prospects",
                "policy_impact": "Monetary policy adjustments affect market liquidity",
                "market_type": "stock",
                "performance": "performing well"
            }
        }
    
    def load_financial_dataset(self, sample_size: int = 10000):
        """加载金融数据集并构建规则系统"""
        try:
            print("Loading financial dataset...")
            self.financial_data = load_dataset("Josephgflowers/Finance-Instruct-500k")
            print(f"Successfully loaded financial dataset with {len(self.financial_data['train'])} training samples")
            
            # 从数据集中提取规则和模板
            self._extract_rules_from_dataset(sample_size)
            
            return True
        except Exception as e:
            print(f"Failed to load financial dataset: {e}")
            return False
    
    def _extract_rules_from_dataset(self, sample_size: int):
        """从数据集中提取规则和模板"""
        print(f"Extracting rules from dataset using {sample_size} samples...")
        
        # 随机采样数据
        total_samples = len(self.financial_data['train'])
        sample_size = min(sample_size, total_samples)
        indices = np.random.choice(total_samples, sample_size, replace=False)
        
        # 提取问题和回答
        questions = []
        responses = []
        
        for idx in indices:
            sample = self.financial_data['train'][idx]
            user_question = sample['user'].strip()
            assistant_response = sample['assistant'].strip()
            
            if user_question and assistant_response:
                questions.append(user_question)
                responses.append(assistant_response)
        
        # 基于数据集内容更新规则
        self._update_rules_from_data(questions, responses)
        
        print(f"Successfully extracted rules from {len(questions)} Q&A pairs")
    
    def _update_rules_from_data(self, questions: List[str], responses: List[str]):
        """基于数据集内容更新规则"""
        # 定义金融领域关键词映射
        financial_categories = {
            'investment': ['invest', 'stock', 'bond', 'fund', 'portfolio', 'asset', 'return', 'risk', 'equity', 'securities'],
            'loan': ['loan', 'mortgage', 'credit', 'debt', 'interest', 'payment', 'borrow', 'lending', 'financing'],
            'insurance': ['insurance', 'premium', 'coverage', 'claim', 'policy', 'life', 'health', 'property', 'liability'],
            'banking': ['bank', 'account', 'deposit', 'withdraw', 'transfer', 'card', 'atm', 'checking', 'savings'],
            'currency': ['currency', 'exchange', 'rate', 'dollar', 'euro', 'yen', 'forex', 'fx', 'foreign exchange'],
            'market': ['market', 'economy', 'economic', 'policy', 'inflation', 'gdp', 'recession', 'growth', 'fiscal'],
            'tax': ['tax', 'taxation', 'deduction', 'income', 'corporate', 'capital gains', 'taxable'],
            'regulation': ['regulation', 'compliance', 'legal', 'law', 'government', 'federal', 'sec', 'federal reserve']
        }
        
        # 为每个类别收集问题和回答
        category_questions = {cat: [] for cat in financial_categories.keys()}
        category_responses = {cat: [] for cat in financial_categories.keys()}
        
        for question, response in zip(questions, responses):
            question_lower = question.lower()
            
            # 找到最匹配的类别
            best_category = None
            max_score = 0
            
            for category, keywords in financial_categories.items():
                score = sum(1 for keyword in keywords if keyword in question_lower)
                if score > max_score:
                    max_score = score
                    best_category = category
            
            if best_category and max_score > 0:
                category_questions[best_category].append(question)
                category_responses[best_category].append(response)
        
        # 更新规则和模板
        self._update_config_from_categories(category_questions, category_responses)
    
    def _update_config_from_categories(self, category_questions: Dict[str, List[str]], category_responses: Dict[str, List[str]]):
        """基于分类结果更新配置"""
        # 更新规则模式
        for category, questions in category_questions.items():
            if questions:
                # 生成模式
                patterns = []
                for question in questions[:10]:  # 使用前10个问题生成模式
                    # 提取问题中的关键模式
                    pattern = self._extract_pattern_from_question(question)
                    if pattern:
                        patterns.append(pattern)
                
                # 更新配置中的规则
                if category in self.config.get("financial_rules", {}):
                    self.config["financial_rules"][category]["patterns"].extend(patterns)
                    self.config["financial_rules"][category]["patterns"] = list(set(self.config["financial_rules"][category]["patterns"]))
                    
                    # 从问题中提取关键词
                    keywords = []
                    for question in questions[:5]:  # 使用前5个问题提取关键词
                        words = question.lower().split()
                        # 过滤掉常见的停用词
                        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'why', 'when', 'where', 'which', 'who'}
                        keywords.extend([word for word in words if word not in stop_words and len(word) > 2])
                    
                    self.config["financial_rules"][category]["keywords"] = list(set(keywords))[:20]  # 限制关键词数量
        
        # 更新回复模板
        for category, responses in category_responses.items():
            if responses:
                # 选择代表性的回答作为模板
                template_responses = []
                for response in responses[:5]:  # 使用前5个回答
                    # 简化回答作为模板
                    template = self._simplify_response_to_template(response)
                    if template:
                        template_responses.append(template)
                
                if template_responses:
                    self.config["response_templates"][category] = template_responses
    
    def _extract_pattern_from_question(self, question: str) -> str:
        """从问题中提取模式"""
        question_lower = question.lower()
        
        # 常见的金融问题模式
        patterns = [
            r'what\s+is\s+([^?]+)\?',
            r'how\s+does\s+([^?]+)\?',
            r'explain\s+([^?]+)',
            r'what\s+are\s+([^?]+)\?',
            r'how\s+to\s+([^?]+)',
            r'difference\s+between\s+([^?]+)',
            r'relationship\s+between\s+([^?]+)',
            r'impact\s+of\s+([^?]+)',
            r'benefits\s+of\s+([^?]+)',
            r'risks\s+of\s+([^?]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question_lower)
            if match:
                return pattern
        
        return None
    
    def _simplify_response_to_template(self, response: str) -> str:
        """将回答简化为模板"""
        # 提取回答中的关键信息
        sentences = response.split('.')
        if sentences:
            # 使用第一个句子作为模板基础
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 50:
                return first_sentence[:100] + "..."
            return first_sentence
        
        return response[:100] + "..." if len(response) > 100 else response
    
    def classify_intent(self, user_input: str) -> Tuple[str, float]:
        """
        分类用户意图
        
        Args:
            user_input: 用户输入
            
        Returns:
            (意图类别, 置信度)
        """
        user_input = user_input.lower()
        max_score = 0
        best_intent = "默认回复"
        
        rules = self.config.get("financial_rules", {})
        
        for intent, rule_data in rules.items():
            patterns = rule_data.get("patterns", [])
            keywords = rule_data.get("keywords", [])
            
            score = 0
            # 模式匹配
            for pattern in patterns:
                if re.search(pattern, user_input):
                    score += 2  # 模式匹配权重更高
            
            # 关键词匹配
            for keyword in keywords:
                if keyword in user_input:
                    score += 1
            
            if score > max_score:
                max_score = score
                best_intent = intent
        
        # 计算置信度
        total_possible_score = len(rules.get(best_intent, {}).get("patterns", [])) * 2 + \
                              len(rules.get(best_intent, {}).get("keywords", []))
        confidence = max_score / total_possible_score if total_possible_score > 0 else 0
        
        return best_intent, confidence
    
    def extract_entities(self, user_input: str) -> Dict[str, str]:
        """
        提取实体信息
        
        Args:
            user_input: 用户输入
            
        Returns:
            提取的实体字典
        """
        entities = {}
        user_input_lower = user_input.lower()
        
        keywords = self.config.get("financial_keywords", {})
        
        # 提取各类实体
        for entity_type, entity_list in keywords.items():
            for entity in entity_list:
                if entity in user_input_lower:
                    entities[entity_type] = entity
                    break
        
        # 提取金额
        amount_patterns = [
            r'(\d+(?:\.\d+)?)[万千百十]?元',
            r'(\d+(?:\.\d+)?)万',
            r'(\d+(?:\.\d+)?)千',
            r'(\d+(?:\.\d+)?)百'
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, user_input)
            if match:
                entities["amount"] = match.group(1)
                break
        
        # 提取利率
        rate_match = re.search(r'(\d+(?:\.\d+)?)%', user_input)
        if rate_match:
            entities["rate"] = rate_match.group(1)
        
        # 提取期限
        term_patterns = [
            r'(\d+)[年个月]',
            r'(\d+)期',
            r'(\d+)天'
        ]
        
        for pattern in term_patterns:
            match = re.search(pattern, user_input)
            if match:
                entities["term"] = match.group(1)
                break
        
        return entities
    
    def generate_rule_based_response(self, intent: str, entities: Dict[str, str]) -> str:
        """
        基于规则生成回复
        
        Args:
            intent: 用户意图
            entities: 提取的实体
            
        Returns:
            生成的回复
        """
        templates = self.config.get("response_templates", {}).get(intent, 
                                                                 self.config.get("response_templates", {}).get("默认回复", ["感谢您的咨询。"]))
        template = random.choice(templates)
        
        # 获取默认值
        default_values = self.config.get("default_values", {})
        
        # 合并实体和默认值
        filled_entities = {**default_values, **entities}
        
        # 填充模板
        try:
            response = template.format(**filled_entities)
        except KeyError as e:
            # 如果模板中需要的实体不存在，使用默认回复
            response = random.choice(self.config.get("response_templates", {}).get("默认回复", ["感谢您的咨询。"]))
        
        return response
    
    def generate_pretrained_response(self, user_input: str) -> str:
        """
        使用预训练模型生成回复
        
        Args:
            user_input: 用户输入
            
        Returns:
            生成的回复
        """
        if not self.use_pretrained:
            return "Sorry, the pretrained model failed to load."
        
        try:
            # 构建对话上下文
            context = ""
            if self.conversation_history:
                recent_history = self.conversation_history[-3:]
                context = " ".join([f"用户: {h['user']} 助手: {h['bot']}" for h in recent_history])

            if self.model_backend == "hf":
                # 编码输入（HF）
                full_input = f"{context} 用户: {user_input} 助手:"
                inputs = self.tokenizer(full_input, return_tensors='pt', padding=True, truncation=True, max_length=512)
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                with torch.no_grad():
                    output = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.shape[1] + 100,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                response = response[len(full_input):].strip()
            else:
                # Reddit 字符级模型推理
                prompt = f"> {user_input}\n>"
                response = self._reddit_generate(prompt=prompt, max_chars=300)

            response = self._clean_response(response)
            return response if response else "Sorry, I cannot generate an appropriate response."
        except Exception as e:
            return f"Error generating response: {e}"

    # ===== Reddit TF 模型相关 =====
    def _init_reddit_model(self, save_dir: str):
        """加载 Reddit 字符级模型检查点"""
        model_path, config_path, vocab_path = self._reddit_get_paths(save_dir)
        with open(config_path, 'rb') as f:
            saved_args = pickle.load(f)
        with open(vocab_path, 'rb') as f:
            chars, vocab = pickle.load(f)

        # 为采样做些参数调整
        self._reddit_beam_width = 2
        saved_args.batch_size = self._reddit_beam_width

        from model import Model  # 复用现有的 Model 定义
        self._reddit_net = Model(saved_args, True)
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self._reddit_sess = tf.Session(config=config)
        tf.global_variables_initializer().run(session=self._reddit_sess)
        saver = tf.train.Saver(self._reddit_net.save_variables_list())
        saver.restore(self._reddit_sess, model_path)
        self._reddit_chars = chars
        self._reddit_vocab = vocab
        self._reddit_states = self._reddit_initial_state()
        # 采样参数
        self._reddit_relevance = -1.0
        self._reddit_temperature = 1.0
        self._reddit_topn = -1

    def _reddit_get_paths(self, input_path: str):
        if os.path.isfile(input_path):
            model_path = input_path
            save_dir = os.path.dirname(model_path)
        elif os.path.exists(input_path):
            save_dir = input_path
            checkpoint = tf.train.get_checkpoint_state(save_dir)
            if checkpoint:
                model_path = checkpoint.model_checkpoint_path
            else:
                # 回退到固定命名
                model_path = os.path.join(save_dir, 'model.ckpt-4735000')
        else:
            raise ValueError('save_dir is not a valid path.')
        return model_path, os.path.join(save_dir, 'config.pkl'), os.path.join(save_dir, 'chars_vocab.pkl')

    def _reddit_initial_state(self):
        return self._reddit_sess.run(self._reddit_net.zero_state)

    def _reddit_forward_char(self, states, char_idx):
        import numpy as np
        shaped_input = np.array([[char_idx]], np.float32)
        inputs = {self._reddit_net.input_data: shaped_input}
        self._reddit_net.add_state_to_feed_dict(inputs, states)
        probs, new_state = self._reddit_sess.run([self._reddit_net.probs, self._reddit_net.final_state], feed_dict=inputs)
        return probs[0], new_state

    def _reddit_sanitize(self, text: str) -> str:
        return ''.join(i for i in text if i in self._reddit_vocab)

    def _reddit_generate(self, prompt: str, max_chars: int = 300) -> str:
        import numpy as np
        # 预热
        states = self._reddit_states
        prime = self._reddit_sanitize(prompt)
        for ch in prime:
            _, states = self._reddit_forward_char(states, self._reddit_vocab[ch])
        # 逐字符生成，直到换行或达到上限
        out_chars = []
        last_idx = self._reddit_vocab.get(' ', 0)
        for _ in range(max_chars):
            probs, states = self._reddit_forward_char(states, last_idx)
            # 温度
            if self._reddit_temperature != 1.0:
                np.seterr(divide='ignore')
                logits = np.log(probs) / max(0.001, self._reddit_temperature)
                logits = logits - np.logaddexp.reduce(logits)
                probs = np.exp(logits)
                np.seterr(divide='warn')
            # 采样
            idx = np.random.choice(len(probs), p=probs / probs.sum())
            ch = self._reddit_chars[idx]
            if ch == '\n':
                break
            out_chars.append(ch)
            last_idx = idx
        return ''.join(out_chars)
    
    def _clean_response(self, response: str) -> str:
        """清理生成的回复"""
        # 移除重复的标点符号
        response = re.sub(r'([。！？])\1+', r'\1', response)
        
        # 限制长度
        if len(response) > 200:
            response = response[:200] + "..."
        
        return response.strip()
    
    def update_conversation_history(self, user_input: str, bot_response: str):
        """更新对话历史"""
        self.conversation_history.append({
            "user": user_input,
            "bot": bot_response
        })
        
        # 保持历史记录在合理范围内
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def find_similar_dataset_question(self, user_input: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """
        在数据集中找到与用户输入最相似的问题
        
        Args:
            user_input: 用户输入
            top_k: 返回前k个最相似的问题
            
        Returns:
            [(问题, 回答, 相似度)] 列表
        """
        if not self.financial_data:
            return []
        
        user_input_lower = user_input.lower()
        similarities = []
        
        # 计算与数据集问题的相似度
        for i in range(min(1000, len(self.financial_data['train']))):  # 限制搜索范围
            sample = self.financial_data['train'][i]
            dataset_question = sample['user'].lower()
            dataset_response = sample['assistant']
            
            # 简单的词汇重叠相似度计算
            user_words = set(user_input_lower.split())
            dataset_words = set(dataset_question.split())
            
            if user_words and dataset_words:
                overlap = len(user_words.intersection(dataset_words))
                similarity = overlap / len(user_words.union(dataset_words))
                
                if similarity > 0.1:  # 只保留相似度大于0.1的结果
                    similarities.append((dataset_question, dataset_response, similarity))
        
        # 按相似度排序并返回前k个
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]
    
    def chat(self, user_input: str, use_pretrained: bool = True, use_dataset: bool = True) -> str:
        """
        主要聊天接口
        
        Args:
            user_input: 用户输入
            use_pretrained: 是否使用预训练模型
            use_dataset: 是否使用数据集匹配
            
        Returns:
            机器人回复
        """
        if not user_input.strip():
            return "Please enter your question."
        
        # 如果启用数据集匹配，先尝试从数据集中找到相似问题
        if use_dataset and self.financial_data:
            similar_questions = self.find_similar_dataset_question(user_input, top_k=1)
            
            if similar_questions:
                dataset_question, dataset_response, similarity = similar_questions[0]
                
                if similarity > 0.3:  # 如果相似度较高，直接返回数据集回答
                    return f"{dataset_response}"
                elif similarity > 0.15:  # 如果相似度中等，返回数据集回答并说明
                    return f"{dataset_response[:1000]}..."
        
        # 分类意图
        intent, confidence = self.classify_intent(user_input)
        # 提取实体
        entities = self.extract_entities(user_input)
        
        # 选择回复方式
        if use_pretrained and self.use_pretrained and confidence < 0.3:
            # 如果规则匹配度低且启用了预训练模型，使用预训练模型
            response = self.generate_pretrained_response(user_input)
        else:
            # 使用基于规则的回复
            print(f"something wrong")

        # 更新对话历史
        self.update_conversation_history(user_input, response)
        
        return response
    
    def get_help(self) -> str:
        """获取帮助信息"""
        return """
        Financial Assistant Usage Instructions:

        Commands:
        - --pretrained: Toggle pretrained model usage
        - --help: Show this help information
        - --reset: Reset conversation history
        - quit/exit: Exit program

        Example Questions:
        - "Explain the difference between fiscal and monetary policy tools used in economics."
        - "Explain the classical economic theory and its policy implications."
        - "Explain how central banks determine currency exchange rates between countries."
        - "Explain how interest rates change with inflation levels in an economy."
        """
    
    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history = []
        print("[Conversation history has been reset]")
    
    def interactive_chat(self):
        """交互式聊天"""
        print("=" * 60)
        print("Welcome to Enhanced Financial Assistant!")
        print("I can provide professional consultation on investment, loans, insurance, banking services and more.")
        print("Type '--help' to view detailed usage instructions")
        print("Type 'quit' or 'exit' to exit the program")
        print("=" * 60)
        
        use_pretrained = True
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("Thank you for using, goodbye!")
                    break
                
                if user_input == '--pretrained':
                    use_pretrained = not use_pretrained
                    status = "enabled" if use_pretrained else "disabled"
                    print(f"[Pretrained model {status}]")
                    continue
                
                if user_input == '--help':
                    print(self.get_help())
                    continue
                
                if user_input == '--reset':
                    self.reset_conversation()
                    continue
                
                if not user_input:
                    continue
                
                response = self.chat(user_input, use_pretrained)
                print(f"Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\nProgram interrupted, goodbye!")
                break
            except Exception as e:
                print(f"Error occurred: {e}")

def main():
    """主函数"""
    # 创建聊天机器人实例
    chatbot = EnhancedFinancialChatbot()
    
    # 加载金融数据集并构建规则系统
    print("Loading financial dataset and building rule system...")
    chatbot.load_financial_dataset(sample_size=100000)  # 使用5000个样本构建规则
    
    # 启动交互式聊天
    chatbot.interactive_chat()

if __name__ == "__main__":
    main()
