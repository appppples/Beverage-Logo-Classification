import markdown
import os
import subprocess

def create_html_and_pdf():
    md_file = r"C:\Users\86449\.gemini\antigravity-ide\brain\4336a428-9dcf-4000-958c-9eb434954937\paper-gemini.md"
    html_file = r"e:\AI Course-Project\Final_Paper.html"
    pdf_file = r"e:\AI Course-Project\《人工智能》课程论文_饮料LOGO分类.pdf"
    
    with open(md_file, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # 替换数据分布截图
    md_text = md_text.replace(
        "> **[在此处插入图片：data_distribution.png]**",
        '<center><img src="file:///e:/AI Course-Project/results/数据预筛与排序.png" style="width:90%; border: 1px solid #ccc;"></center>'
    )

    # 替换CNN曲线和训练截图
    md_text = md_text.replace(
        "> **[在此处插入图片：SimpleCNN_loss_curve.png 与 SimpleCNN_acc_curve.png]**",
        '''<center>
        <img src="file:///e:/AI Course-Project/results/SimpleCNN_acc_curve.png" style="width:48%;"> 
        <img src="file:///e:/AI Course-Project/results/SimpleCNN_loss_curve.png" style="width:48%;">
        <br><br>
        <img src="file:///e:/AI Course-Project/results/训练截图CNN.png" style="width:90%; border: 1px solid #ccc;">
        </center>'''
    )

    # 替换ResNet曲线和训练截图
    md_text = md_text.replace(
        "> **[在此处插入图片：ResNet50_loss_curve.png 与 ResNet50_acc_curve.png]**",
        '''<center>
        <img src="file:///e:/AI Course-Project/results/ResNet50_acc_curve.png" style="width:48%;"> 
        <img src="file:///e:/AI Course-Project/results/ResNet50_loss_curve.png" style="width:48%;">
        <br><br>
        <img src="file:///e:/AI Course-Project/results/训练截图resnet.png" style="width:90%; border: 1px solid #ccc;">
        </center>'''
    )

    # 替换ROC
    md_text = md_text.replace(
        "> **[在此处插入图片：SimpleCNN_roc_curve.png 与 ResNet50_roc_curve.png]**",
        '''<center>
        <img src="file:///e:/AI Course-Project/results/SimpleCNN_roc_curve.png" style="width:48%;"> 
        <img src="file:///e:/AI Course-Project/results/ResNet50_roc_curve.png" style="width:48%;">
        </center>'''
    )

    # 替换混淆矩阵
    md_text = md_text.replace(
        "> **[在此处插入图片：ResNet50_confusion_matrix.png]**",
        '<center><img src="file:///e:/AI Course-Project/results/ResNet50_confusion_matrix.png" style="width:70%;"></center>'
    )

    # 替换错误案例
    md_text = md_text.replace(
        "> **[在此处插入图片：ResNet50_error_examples.png]**",
        '<center><img src="file:///e:/AI Course-Project/results/ResNet50_error_examples.png" style="width:90%;"></center>'
    )

    # 增加GUI测试章节
    gui_section = '''
### (六) GUI 图形化界面测试
为了更直观地验证模型的泛化能力与实际应用价值，本文基于 Tkinter 构建了双模型对比测试客户端。可以直接拖入互联网上任意寻找的新图片进行预测。从测试截图可以看出，即使在面临反光、遮挡和角度倾斜的复杂自然场景下，ResNet50 仍能以极高的置信度给出正确预测，而基线 SimpleCNN 往往表现挣扎。
<center>
<img src="file:///e:/AI Course-Project/results/屏幕截图 2026-06-15 115703.png" style="width:48%; border: 1px solid #ccc; margin-bottom: 10px;"> 
<img src="file:///e:/AI Course-Project/results/屏幕截图 2026-06-15 115753.png" style="width:48%; border: 1px solid #ccc; margin-bottom: 10px;">
<img src="file:///e:/AI Course-Project/results/屏幕截图 2026-06-15 115804.png" style="width:48%; border: 1px solid #ccc;">
</center>
<br>
'''
    md_text = md_text.replace("## 7. 结论", gui_section + "\n## 7. 结论")

    html_content = markdown.markdown(md_text, extensions=['tables'])

    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        @page {{ size: A4; margin: 2.54cm 3.18cm; }}
        body {{
            font-family: 'SimSun', serif;
            font-size: 16px; /* 小四 */
            line-height: 1.5;
            color: #000;
        }}
        h1, h2, h3 {{
            font-family: 'SimHei', sans-serif;
            margin-top: 1.5em;
        }}
        h1 {{ font-size: 24px; text-align: center; }}
        h2 {{ font-size: 20px; }}
        h3 {{ font-size: 18px; }}
        p {{ text-indent: 2em; margin-top: 0.5em; margin-bottom: 0.5em; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-family: 'Times New Roman', SimSun; font-size: 14px; }}
        th, td {{ border: 1px solid black; padding: 6px; text-align: center; }}
        .center {{ text-align: center; font-size: 14px; margin-top: 5px; }}
        blockquote {{ margin-left: 0; padding-left: 1em; border-left: 4px solid #ccc; font-style: italic; color: #555; }}
    </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(styled_html)
        
    print(f"Generated HTML at {html_file}")
    
    # Use Microsoft Edge to print to PDF
    edge_paths = [
        r"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
        r"C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe"
    ]
    edge_exe = None
    for p in edge_paths:
        if os.path.exists(p):
            edge_exe = p
            break
            
    if edge_exe:
        print(f"Using Edge to generate PDF: {edge_exe}")
        cmd = f'"{edge_exe}" --headless --print-to-pdf="{pdf_file}" "{html_file}"'
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"Successfully generated PDF at {pdf_file}")
        except Exception as e:
            print(f"Failed to generate PDF: {e}")
    else:
        print("Microsoft Edge not found. Open the HTML file in any browser and 'Print to PDF'.")

if __name__ == "__main__":
    create_html_and_pdf()
