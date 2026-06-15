import markdown
import pdfkit
import os

def convert_md_to_pdf(md_path, pdf_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
        
    html = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    
    # Add styling for PDF
    styled_html = f"""
    <html>
    <head>
    <style>
        body {{
            font-family: 'SimSun', serif; /* 小四号对应宋体 */
            font-size: 12pt; /* 小四号 */
            line-height: 1.5;
            padding: 2cm;
        }}
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'SimHei', sans-serif;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }}
        img {{
            max-width: 100%;
            display: block;
            margin: 0 auto;
        }}
        .center {{
            text-align: center;
            font-size: 10pt; /* 图标题小一号 */
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
            font-family: 'Times New Roman', serif; /* 英文和数字用新罗马 */
            font-size: 11pt; /* 表格字体小一号 */
        }}
        th, td {{
            border: 1px solid black;
            padding: 8px;
            text-align: center;
        }}
        /* 公式放表格，表格外框隐去 */
        .formula-table, .formula-table th, .formula-table td {{
            border: none;
            width: 100%;
        }}
        code, pre {{
            font-family: 'Courier New', monospace;
            background-color: #f5f5f5;
            padding: 2px 4px;
        }}
        pre {{
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }}
    </style>
    </head>
    <body>
    {html}
    </body>
    </html>
    """
    
    options = {
        'page-size': 'A4',
        'margin-top': '25.4mm',
        'margin-right': '31.8mm',
        'margin-bottom': '25.4mm',
        'margin-left': '31.8mm',
        'encoding': 'UTF-8',
    }
    
    try:
        # Note: wkhtmltopdf must be installed on the system and in PATH
        pdfkit.from_string(styled_html, pdf_path, options=options)
        print(f"Successfully generated PDF: {pdf_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print("Note: You may need to install wkhtmltopdf from https://wkhtmltopdf.org/downloads.html")

if __name__ == "__main__":
    md_file = r"C:\Users\86449\.gemini\antigravity-ide\brain\4336a428-9dcf-4000-958c-9eb434954937\paper-gemini.md"
    pdf_file = "e:\\AI Course-Project\\Final_Paper.pdf"
    
    if os.path.exists(md_file):
        convert_md_to_pdf(md_file, pdf_file)
    else:
        print(f"Markdown file not found: {md_file}")
