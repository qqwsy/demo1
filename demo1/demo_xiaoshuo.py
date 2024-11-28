from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.agents import Tool, initialize_agent
from modelscope import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model_name = "qwen/Qwen2.5-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def generate_with_qwen(prompt, max_length=512, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length, temperature=temperature)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_story_outline(chapter_title: str) -> str:
    prompt = f"为小说章节 '{chapter_title}' 生成详细的剧情，包括冲突、高潮和结尾："
    return generate_with_qwen(prompt)


def create_character(name: str) -> str:
    prompt = f"创建一个名为{name}的小说角色，描述其性格特征、背景故事和与主线情节的关联："
    return generate_with_qwen(prompt)


def polish_text(text: str) -> str:
    prompt = f"以下是小说片段，请优化语言，使其更生动流畅，并确保语法无误：\n{text}"
    return generate_with_qwen(prompt)


story_agent_tool = Tool(
    name="StoryAgent",
    func=generate_story_outline,
    description="生成小说章节剧情内容"
)
character_agent_tool = Tool(
    name="CharacterAgent",
    func=create_character,
    description="生成角色设定"
)
polish_agent_tool = Tool(
    name="PolishAgent",
    func=polish_text,
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
    characters = ["主角阿尔法", "反派赫德", "AI盟友塞莉娜"]
    for char in characters:
        character_info = novel_agent.run(f"生成角色: {char}")
        print(f"角色生成: {character_info}")

    # 润色内容
    polished_text = novel_agent.run(f"润色文本: {story_outline}")
    print(f"完成章节: {polished_text}")
