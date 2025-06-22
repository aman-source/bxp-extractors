from weasyprint import HTML

# HTML content (can also be a file or URL)
html_content = """
<html>
  <head>
    <title>Sample PDF</title>
  </head>
  <body>
    <h1>Hello, PDF!</h1>
    <p>This PDF was generated from HTML using WeasyPrint.</p>
  </body>
</html>
"""

HTML(string=html_content).write_pdf("output.pdf")
print("PDF generated: output.pdf")
