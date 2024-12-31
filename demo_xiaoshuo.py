from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.agents import Tool, initialize_agent
import torch

# 加载模型和分词器
model_name = "qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else "auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 集合用于存储已生成的内容，防止重复
generated_contents = set()


def generate_with_qwen(prompt, target_word_count=1000, tolerance=100, temperature=0.7, top_k=50, top_p=0.95,
                       repetition_penalty=1.2, no_repeat_ngram_size=3):
    # 估算tokens数，对于中文，1字 ≈ 1 token
    max_length = target_word_count + tolerance
    min_length = target_word_count - tolerance

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        min_length=min_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=True,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 后处理：根据实际字数进行截断或补充
    actual_word_count = len(generated_text)

    if actual_word_count > target_word_count + tolerance:
        # 截断到目标字数
        generated_text = generated_text[:target_word_count]
    elif actual_word_count < target_word_count - tolerance:
        # 补充生成内容
        additional_text = generate_with_qwen(prompt, target_word_count - actual_word_count, tolerance, temperature,
                                             top_k, top_p, repetition_penalty, no_repeat_ngram_size)
        generated_text += additional_text

    # 检查是否重复
    if generated_text in generated_contents:
        return "生成内容重复，请尝试调整提示或参数。"
    generated_contents.add(generated_text)
    return generated_text


def generate_unique_content(prompt, target_word_count=1000, max_attempts=5):
    for _ in range(max_attempts):
        content = generate_with_qwen(prompt, target_word_count)
        if content not in generated_contents:
            generated_contents.add(content)
            return content
    return "未能生成唯一的内容，请尝试调整提示或参数。"


def generate_story_outline(chapter_title: str, word_count: int = 2000) -> str:
    prompt = (
        f"为小说章节 '{chapter_title}' 设计一段精彩且独特的剧情，包含以下内容：主要冲突、关键转折、高潮以及结尾。"
        "请确保情节紧凑，人物动机清晰，情感表达真实，且各部分内容互不重复，符合章节的整体氛围。"
    )
    return generate_unique_content(prompt, target_word_count=word_count)


def create_character(name: str, word_count: int = 800) -> str:
    prompt = (
        f"为小说创作一个名为{name}的角色。请详细描述该角色的性格特征、外貌、成长背景，以及他/她与主线情节的关系。"
        "请特别强调角色的内心冲突和在故事中的成长轨迹，确保描述内容与已有角色设定不重复。"
    )
    return generate_unique_content(prompt, target_word_count=word_count)


def polish_text(text: str, word_count: int = 0) -> str:
    prompt = (
        f"请对以下小说片段进行润色，提升语言的生动性和流畅度，确保语法无误。"
        "请特别注意修饰词的使用，使描写更具画面感，并优化句式结构以增强情感表达，同时避免重复内容：\n{text}"
    )
    if word_count > 0:
        return generate_unique_content(prompt, target_word_count=word_count)
    else:
        return generate_unique_content(prompt)


# 定义工具
story_agent_tool = Tool(
    name="StoryAgent",
    func=lambda input: generate_story_outline(input),
    description="生成小说章节剧情内容"
)
character_agent_tool = Tool(
    name="CharacterAgent",
    func=lambda input: create_character(input),
    description="生成角色设定"
)
polish_agent_tool = Tool(
    name="PolishAgent",
    func=lambda input: polish_text(input),
    description="润色文本"
)

tools = [story_agent_tool, character_agent_tool, polish_agent_tool]
novel_agent = initialize_agent(tools, llm=None, agent="zero-shot-react-description", verbose=True)

chapters = ["序章：初见未来", "第一章：虚拟与现实", "第二章：深渊的秘密"]

# 执行生成
for chapter in chapters:
    print(f"正在生成章节: {chapter}")

    # 剧情生成
    story_outline = novel_agent.run(f"生成剧情: {chapter}")

    # 为章节创建角色
    characters = ["主角盖伦", "反派德莱厄斯", "盟友加里奥"]
    for char in characters:
        character_info = novel_agent.run(f"生成角色: {char}")
        print(f"角色生成: {character_info}")

    # 润色内容
    polished_text = novel_agent.run(f"润色文本: {story_outline}")
    print(f"完成章节: {polished_text}\n")
