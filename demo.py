# 1，第一步文件导入+数据清洗
import pandas as pd

# 读取Excel文件（请将路径替换为实际文件路径）然后赋值
file_path = "电商数据.xlsx"
# 返回二维表格数据结构赋值给df
df = pd.read_excel(file_path)

# 筛选需要分析的字段
keep_columns = ['商品名', '商家实收金额', '商品数量']
# 先检查列是否存在，避免KeyError
existing_cols = [col for col in keep_columns if col in df.columns]
if len(existing_cols) != len(keep_columns):
    missing = set(keep_columns) - set(existing_cols)
    print(f"警告：以下列不存在，将被忽略：{missing}")

df_filtered = df[existing_cols].copy()

# 3. 去空值（删除任何包含空值的行）
df_cleaned = df_filtered.dropna()

# 4. 打印清洗后的数据
print("清洗后的数据预览：")
print(df_cleaned.to_string(index=False))  # 不显示行索引，完整打印

# 第二步：将清洗后的数据发送给大模型进行分析

from openai import OpenAI  # 使用 OpenAI SDK 调用 DeepSeek（兼容）

# ---------- 2.1 准备数据摘要（避免发送全量原始数据，节省 Token）----------
def prepare_summary(df_cleaned):
    """
    对清洗后的数据进行聚合统计，生成大模型易于理解的文本摘要。
    """
    # 按商品名分组统计
    summary = df_cleaned.groupby('商品名').agg({
        '商品数量': 'sum',
        '商家实收金额': 'sum'
    }).rename(columns={
        '商品数量': '总销量',
        '商家实收金额': '总销售额'
    }).sort_values('总销售额', ascending=False)

    total_sales = summary['总销售额'].sum()
    total_quantity = summary['总销量'].sum()
    avg_price = total_sales / total_quantity if total_quantity > 0 else 0

    # 构建结构化的数据文本
    data_text = f"""
【整体运营数据】
- 有效订单数：{len(df_cleaned)} 笔
- 总销售额：{total_sales:.2f} 元
- 总销量：{total_quantity} 件
- 平均客单价：{avg_price:.2f} 元

【商品销售排行榜】（前10名）
{summary.head(10).to_string()}

【全量商品销售明细】
{summary.to_string()}
"""
    return data_text

# ---------- 2.2 调用 qvq 大模型进行分析 ----------
def analyze_with_qvq(data_text, api_key):
    """
    使用 DeepSeek API 进行智能分析
    API 文档：https://api-docs.deepseek.com/
    """
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # qvq 官方 endpoint
    )

    prompt = f"""
你是一位经验丰富的电商运营专家。请严格基于以下销售数据，完成三项分析任务：

1. 【销量分析】：整体销售表现如何？哪个商品贡献最大？是否存在销售集中现象？
2. 【爆款推荐】：推荐3款最具潜力的爆款商品，并分别给出具体推荐理由。
3. 【用户偏好总结】：根据商品名称中的关键词（如“石墨烯”、“墙暖”、“取暖神器”等），总结用户的核心需求与购买偏好。

请用清晰的小标题分段输出，语言专业、简洁。

=== 销售数据如下 ===
{data_text}
=== 数据结束 ===
"""

    try:
        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": "你是一位严谨的电商数据分析师，只基于数据给出结论。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500,
            stream=True
        )
        # 逐块读取流式内容
        full_response = ""
        print("🤖 模型分析中...\n")
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)  # 实时显示
                full_response += content
        return full_response
    except Exception as e:
        print(f"❌ API 调用失败：{e}")
        return None


# ---------- 2.3 主执行逻辑（直接放在第二步末尾，会自动运行）----------
if __name__ == "__main__":
    # 假设前面的代码已经定义了 df_cleaned，这里直接使用
    # 如果 df_cleaned 未定义，请确保第一步的代码先执行。

    # 1. 生成数据摘要
    summary_text = prepare_summary(df_cleaned)
    print("\n📊 正在将以下数据摘要发送给大模型...\n")
    print(summary_text)

    # 2. 设置你的 qvq API Key（请替换为真实 Key）
    DEEPSEEK_API_KEY = "sk-a884fa1d0a2e4f20a0e7e84c09a6210e"

    if DEEPSEEK_API_KEY == "sk-a884fa1d0a2e4f20a0e7e84c09a6210e":
        print("\n🤖 正在调用 qvq 进行分析，请稍候...")
        print("\n" + "=" * 50)
        print("🎯 大模型分析结论")
        print("=" * 50)
        analysis_result = analyze_with_qvq(summary_text, DEEPSEEK_API_KEY)
        if analysis_result:
            with open("销售分析结果.txt", "w", encoding="utf-8") as f:
                f.write(analysis_result)
                print("\n\n✅ 分析完成！结果已保存到：销售分析结果.txt")
        else:
            print("⚠️ 分析失败，请检查 API Key 或网络。")
    else:
        print("\n⚠️ 未设置有效的 DeepSeek API Key，跳过分析步骤。")
        print("👉 获取 Key：https://bailian.console.aliyun.com/cn-beijing?tab=model#/api-key")
