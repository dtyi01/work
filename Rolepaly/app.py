import os
import json
import torch
import argparse
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from pathlib import Path


class RoleManager:
    """
    角色配置管理器，负责加载和解析角色设定文件
    
    支持从config目录读取JSON格式的角色配置，并提供默认角色回退机制。
    """

    DEFAULT_ROLE = {
        "role_name": "default_assistant",
        "personality": "友好且乐于助人，回答简洁明了",
        "knowledge_base": ["通用知识", "基础科学", "日常技能"],
        "dialogue_style": {
            "sentence_length": "中等",
            "tone": "平和友善",
            "taboo_words": []
        },
        "catchphrases": ["没问题！", "让我想想...", "这是个好问题"]
    }

    def __init__(self, config_dir: str = "config"):
        """
        初始化角色管理器
        
        Args:
            config_dir (str): 角色配置文件存储目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

    def load_role(self, role_name: str) -> Dict:
        """
        加载指定角色的配置
        
        Args:
            role_name (str): 角色名称
            
        Returns:
            Dict: 角色配置字典，加载失败时返回默认配置
        """
        config_path = self.config_dir / f"{role_name}.json"
        
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    role_config = json.load(f)
                print(f"✅ 成功加载角色配置: {role_name}")
                return role_config
            else:
                print(f"⚠️  角色配置文件未找到: {config_path}，使用默认配置")
                return self.DEFAULT_ROLE.copy()
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"❌ 加载角色配置失败: {e}，使用默认配置")
            return self.DEFAULT_ROLE.copy()

    def list_available_roles(self) -> List[str]:
        """
        列出所有可用的角色配置文件
        
        Returns:
            List[str]: 可用角色名称列表
        """
        roles = []
        if self.config_dir.exists():
            for file in self.config_dir.glob("*.json"):
                roles.append(file.stem)
        return sorted(roles)


class MemoryManager:
    """
    对话历史记忆管理器，实现持久化存储
    
    使用JSON文件为每个角色独立保存对话历史，支持加载、保存和更新操作。
    """

    def __init__(self, memory_dir: str = "memory"):
        """
        初始化记忆管理器
        
        Args:
            memory_dir (str): 记忆存储目录
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.current_role = None
        self.history = []

    def load_history(self, role_name: str) -> List[Dict]:
        """
        加载指定角色的对话历史
        
        Args:
            role_name (str): 角色名称
            
        Returns:
            List[Dict]: 对话历史记录列表
        """
        self.current_role = role_name
        memory_path = self.memory_dir / f"{role_name}_history.json"
        
        try:
            if memory_path.exists():
                with open(memory_path, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                print(f"✅ 已加载 {len(self.history)} 条历史消息")
            else:
                # 创建初始系统消息
                system_msg = {"role": "system", "content": "你是一个AI助手"}
                self.history = [system_msg]
                self.save_history()
                print("📝 新对话历史已创建")
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"❌ 读取历史记录失败: {e}，创建新记录")
            self.history = [{"role": "system", "content": "你是一个AI助手"}]
            
        return self.history.copy()

    def save_history(self):
        """保存当前对话历史到文件"""
        if not self.current_role:
            return
            
        memory_path = self.memory_dir / f"{self.current_role}_history.json"
        
        try:
            with open(memory_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"❌ 无法保存历史记录: {e}")

    def update_history(self, user_input: str, assistant_response: str):
        """
        更新对话历史并保存
        
        Args:
            user_input (str): 用户输入内容
            assistant_response (str): 助手回复内容
        """
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": assistant_response})
        self.save_history()


class ContextManager:
    """
    上下文感知管理器，控制对话历史长度
    
    实现滑动窗口机制，动态裁剪对话历史以适应模型的上下文限制（最大8192 tokens）。
    """

    MAX_CONTEXT_TOKENS = 8192
    RESPONSE_BUFFER = 1024  # 预留空间给模型生成回复

    def __init__(self, tokenizer):
        """
        初始化上下文管理器
        
        Args:
            tokenizer: Hugging Face 分词器实例
        """
        self.tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量
        
        Args:
            text (str): 输入文本
            
        Returns:
            int: token数量
        """
        return len(self.tokenizer.encode(text))

    def build_context_prompt(self, history: List[Dict], current_query: str, role_config: Dict) -> str:
        """
        构建包含上下文的完整prompt
        
        Args:
            history (List[Dict]): 对话历史记录
            current_query (str): 当前用户查询
            role_config (Dict): 角色配置
            
        Returns:
            str: 格式化的prompt字符串
        """
        # 构建系统提示
        personality_desc = role_config.get("personality", "")
        style_info = role_config.get("dialogue_style", {})
        tone_desc = style_info.get("tone", "自然")
        knowledge = ", ".join(role_config.get("knowledge_base", []))
        
        system_prompt = (
            f"你正在扮演{role_config['role_name']}。你的性格是{personality_desc}，"
            f"说话风格具有{tone_desc}的特点，知识领域涵盖{knowledge}。"
        )
        
        # 添加口头禅约束
        catchphrases = role_config.get("catchphrases", [])
        if catchphrases:
            system_prompt += f"你可以适时使用这些口头禅：{', '.join(catchphrases)}。"
        
        # 构建完整对话序列
        prompt_parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
        
        # 添加历史消息（应用滑动窗口）
        trimmed_history = self._apply_sliding_window(history[1:], current_query, role_config)
        for msg in trimmed_history:
            role_tag = "user" if msg["role"] == "user" else "assistant"
            prompt_parts.append(f"<|im_start|>{role_tag}\n{msg['content']}<|im_end|>")
        
        # 添加当前查询
        prompt_parts.append(f"<|im_start|>user\n{current_query}<|im_end|>\n<|im_start|>assistant\n")
        
        return "\n".join(prompt_parts)

    def _apply_sliding_window(self, history: List[Dict], new_query: str, role_config: Dict) -> List[Dict]:
        """
        应用滑动窗口裁剪策略，确保总token数不超过限制
        
        Args:
            history (List[Dict]): 原始对话历史
            new_query (str): 新的用户查询
            role_config (Dict): 角色配置
            
        Returns:
            List[Dict]: 裁剪后的对话历史
        """
        if not history:
            return []
            
        max_input_tokens = self.MAX_CONTEXT_TOKENS - self.RESPONSE_BUFFER
        
        # 计算系统提示和新查询的token数
        system_prompt = f"你正在扮演{role_config['role_name']}。"
        base_tokens = (
            self.count_tokens(system_prompt) +
            self.count_tokens(new_query) +
            50  # 安全余量
        )
        
        available_tokens = max_input_tokens - base_tokens
        if available_tokens <= 0:
            print("⚠️  上下文空间不足，仅保留最近一轮对话")
            return history[-2:] if len(history) >= 2 else []
        
        # 从最近的对话开始反向累加，直到超出预算
        cumulative_tokens = 0
        selected_messages = []
        
        for i in range(len(history) - 1, -1, -1):
            msg_tokens = self.count_tokens(history[i]["content"]) + 20  # 包含标签开销
            if cumulative_tokens + msg_tokens > available_tokens:
                break
            selected_messages.append(history[i])
            cumulative_tokens += msg_tokens
        
        # 恢复原始顺序
        return list(reversed(selected_messages))


class StreamOutput:
    """
    流式输出处理器，实现逐字生成效果
    
    使用TextIteratorStreamer实现真正的token级流式输出，支持可调节的显示速度。
    """

    SPEED_PRESETS = {
        'slow': 0.1,
        'normal': 0.03,
        'fast': 0.01,
        'ultra_fast': 0.001
    }

    def __init__(self, speed: str = 'normal'):
        """
        初始化流式输出器
        
        Args:
            speed (str): 输出速度预设 ('slow', 'normal', 'fast', 'ultra_fast')
        """
        self.speed = self.SPEED_PRESETS.get(speed, 0.03)

    def stream_print(self, streamer):
        """
        执行流式打印输出
        
        Args:
            streamer: TextIteratorStreamer实例
        """
        print("Assistant: ", end="", flush=True)
        for new_text in streamer:
            if new_text and not new_text.isspace():
                for char in new_text:
                    print(char, end="", flush=True)
                    # time.sleep(self.speed)  # 逐字符延迟已在streamer层面处理
        print()  # 最终换行


class QwenAgent:
    """
    Qwen-1.8B-Chat 离线AI Agent主类
    
    集成角色管理、记忆存储、上下文控制和流式输出功能，提供完整的对话体验。
    """

    def __init__(self, role_name: str = "default_assistant", output_speed: str = "normal"):
        """
        初始化AI Agent
        
        Args:
            role_name (str): 初始角色名称
            output_speed (str): 输出速度设置
        """
        self.role_manager = RoleManager()
        self.memory_manager = MemoryManager()
        self.role_config = self.role_manager.load_role(role_name)
        self.output_handler = StreamOutput(output_speed)
        
        # 自动检测设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎮 设备检测: {'GPU' if self.device == 'cuda' else 'CPU'}")
        
        # 加载模型和分词器
        self.model_name = "Qwen/Qwen-1.8B-Chat"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).eval()
        
        # 初始化上下文管理器
        self.context_manager = ContextManager(self.tokenizer)
        
        # 加载对话历史
        self.history = self.memory_manager.load_history(role_name)
        print(f"🚀 AI Agent已启动，当前角色: {self.role_config['role_name']}")

    def generate_response(self, user_input: str):
        """
        生成助手响应的核心方法
        
        Args:
            user_input (str): 用户输入文本
        """
        try:
            # 构建上下文prompt
            prompt = self.context_manager.build_context_prompt(
                self.history, user_input, self.role_config
            )
            
            # 对于长对话，使用流式生成
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # 在子线程中执行模型推理
            gen_kwargs = {
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.8,
                "do_sample": True,
                "streamer": streamer
            }
            
            thread = Thread(target=self.model.generate, kwargs={**inputs, **gen_kwargs})
            thread.start()
            
            # 流式输出结果
            self.output_handler.stream_print(streamer)
            
            # 获取完整响应（用于保存）
            full_response = ""
            for new_text in streamer:
                full_response += new_text
                
            # 清理响应
            clean_response = full_response.strip()
            
            # 更新记忆
            self.memory_manager.update_history(user_input, clean_response)
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": clean_response})
            
        except Exception as e:
            error_msg = f"❌ 推理过程中发生错误: {str(e)}"
            print(error_msg)
            fallback_response = "抱歉，我暂时无法处理这个问题，请稍后再试。"
            print(f"Assistant: {fallback_response}")
            self.memory_manager.update_history(user_input, fallback_response)


def main():
    """
    主函数：解析命令行参数并启动AI Agent
    """
    parser = argparse.ArgumentParser(description="Qwen-1.8B-Chat 离线AI Agent")
    parser.add_argument("--role", type=str, default="default_assistant", help="指定角色名称")
    parser.add_argument("--speed", type=str, choices=['slow', 'normal', 'fast', 'ultra_fast'], 
                       default='normal', help="输出速度 (slow/normal/fast/ultra_fast)")
    parser.add_argument("--list-roles", action="store_true", help="列出所有可用角色")
    
    args = parser.parse_args()
    
    # 如果请求列出角色，则只显示可用角色
    if args.list_roles:
        role_manager = RoleManager()
        roles = role_manager.list_available_roles()
        print("可用角色列表:")
        for role in roles:
            config = role_manager.load_role(role)
            desc = config.get("personality", "无描述")
            print(f"  • {role}: {desc}")
        return
    
    # 启动AI Agent
    agent = QwenAgent(role_name=args.role, output_speed=args.speed)
    
    # 对话主循环
    print("\n开始对话 (输入 '退出' 或 'quit' 结束):")
    while True:
        try:
            user_input = input("\nUser: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['退出', 'quit', 'exit']:
                print("再见！")
                break
                
            agent.generate_response(user_input)
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            continue


if __name__ == "__main__":
    main()
