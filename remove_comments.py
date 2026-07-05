import os
import re
import tokenize
import io

def remove_python_comments(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
                                                                         
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        modified_tokens = []
        for token in tokens:
            if token.type == tokenize.COMMENT:
                              
                if token.string.startswith('#!'):
                    modified_tokens.append(token)
                else:
                                                                                          
                    continue
            else:
                modified_tokens.append(token)
                
        result = tokenize.untokenize(modified_tokens)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Removed comments from Python file: {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def remove_js_css_comments(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
                                         
        content = re.sub(r'/\*[\s\S]*?\*/', '', content)
        
                                                                            
                                                               
        content = re.sub(r'(?<!:)//.*', '', content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Removed comments from JS/CSS file: {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def remove_html_comments(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
                                           
        content = re.sub(r'<!--[\s\S]*?-->', '', content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Removed comments from HTML file: {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    root_dir = r"c:\Users\PANDA\Skye"
    exclude_dirs = {'.git', 'node_modules', 'dist', 'build', '.vite', '__pycache__', '.agents'}
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
                                         
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            ext = os.path.splitext(filename)[1].lower()
            
            if filepath == __file__:
                continue
                
            if ext == '.py':
                remove_python_comments(filepath)
            elif ext in ('.js', '.jsx', '.ts', '.tsx', '.css'):
                remove_js_css_comments(filepath)
            elif ext in ('.html', '.htm'):
                remove_html_comments(filepath)

if __name__ == '__main__':
    main()
