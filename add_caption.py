from PIL import Image, ImageDraw, ImageFont, ImageFilter
import textwrap

def add_news_caption(image_path, english_text, chinese_text, output_path):
    # 加载图片
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    english_font_path = "./dejavu-fonts-ttf-2.37/ttf/DejaVuSans-Bold.ttf"
    chinese_font_path = "./SubsetOTF/CN/SourceHanSansCN-Bold.otf"

    # 设置字体和大小
    # image.resize((542, 420))
    image_width, image_height = image.size
    print(image.size)

    # 初始字体大小
    max_title_font_size = int(image_width * 0.040)
    max_subtitle_font_size = int(image_width * 0.040)

    # 尝试动态字体大小
    title_font_size = max_title_font_size
    subtitle_font_size = max_subtitle_font_size

    while title_font_size > 10:
        title_font = ImageFont.truetype(english_font_path, title_font_size)
        subtitle_font = ImageFont.truetype(chinese_font_path, subtitle_font_size)

        eng_lines = textwrap.wrap(english_text, width=40)
        chi_lines = textwrap.wrap(chinese_text, width=20)

        # 估计高度
        total_text_height = (len(eng_lines) * (title_font_size + 10)) + 10 + (len(chi_lines) * (subtitle_font_size + 10))

        if total_text_height > image_height * 0.20:
            title_font_size -= 2
            subtitle_font_size -= 2
        else:
            break

    title_font = ImageFont.truetype(english_font_path, title_font_size)
    subtitle_font = ImageFont.truetype(chinese_font_path, subtitle_font_size)

    # 重新计算行
    eng_lines = textwrap.wrap(english_text, width=65)
    chi_lines = textwrap.wrap(chinese_text, width=35)

    # 计算字幕条高度
    bar_padding = 0
    total_text_height = (len(eng_lines) * (title_font_size + 10)) + 10 + (len(chi_lines) * (subtitle_font_size + 10))
    bar_height = total_text_height + bar_padding * 2 - 40

    # 绘制白色半透明背景
    overlay = Image.new('RGBA', (image_width, bar_height), (255, 255, 255, 230))
    image.paste(overlay, (0, image_height - bar_height), overlay)

    # 红色底线
    line_height = 5
    draw.rectangle(
        [(0, image_height - line_height), (image_width, image_height)],
        fill=(200, 0, 0)
    )

    # 开始绘制文字
    current_y = image_height - bar_height + bar_padding

    text_color = (0, 0, 0)

    blur_crop = image.crop((0, image_height - bar_height, image_width, image_height)).filter(ImageFilter.GaussianBlur(radius=10))
    image.paste(blur_crop, (0, image_height - bar_height))
    text_color = (0, 0, 0)

    # 绘制英文每一行
    for line in eng_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        line_width = bbox[2] - bbox[0]
        x = (image_width - line_width) // 2
        draw.text((x, current_y), line, font=title_font, fill=text_color)
        current_y += title_font_size   # 每行加行距

    # 英文和中文之间额外留一点距离
    current_y += 0

    # 绘制中文每一行
    for line in chi_lines:
        bbox = draw.textbbox((0, 0), line, font=subtitle_font)
        line_width = bbox[2] - bbox[0]
        x = (image_width - line_width) // 2
        draw.text((x, current_y), line, font=subtitle_font, fill=text_color)
        current_y += subtitle_font_size

    # 保存图片
    image.save(output_path)
    print(f"字幕图像已保存至: {output_path}")

# 示例调用
add_news_caption(
    "./example/850.jpg",
    # "Former Congressman Accused of Using Secret Government Funds to Purchase Rare Coin Collection, Family Involved",
    # "前国会议员被指控使用秘密政府资金购买稀有硬币收藏，其家人也参与其中",
    # "Secret White House recordings reveal that JFK allegedly disclosed plans to overturn election results using classified data",
    # "白宫秘密录音揭露肯尼迪总统涉嫌利用机密数据推翻选举结果",
    # "Kyle Taylor left and Antony Evans stand outside their shipping container lab in San Francisco.",
    # "Secretary of State John Kerry leads a fashion show in Riyadh, Saudi Arabia, showcasing traditional attire.",
    "Fox News Host Accused of Secretly Funding Illegal Border Wall Construction, Leaked Documents Reveal",
    "时任众议院少数党党鞭的坎托在 NBC 的《会见新闻界》节目中接受戴维·格雷戈里的采访，坎托在节目中谈论奥巴马政府的经济复苏计划",
    # "约翰·克里国务卿在沙特阿拉伯利雅得主持时装秀，展示传统服饰",
    "captioned.jpg"
)