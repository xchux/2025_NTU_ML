
import json
import re

nb_path = '/workspace/Homework_8_Model_Editing.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Implement apply_rome_to_model update matrix
def fix_apply_rome_source(source_list):
    new_source = []
    has_fix = False
    for line in source_list:
        if 'delta_v.T @ delta_u.T' in line:
            has_fix = True
            
    if has_fix: return source_list

    for line in source_list:
        if '# upd_matrix = ...@...' in line:
            new_source.append(line)
            new_source.append('                delta_u, delta_v = deltas[w_name]\n')
            new_source.append('                upd_matrix = delta_v.T @ delta_u.T\n')
        else:
            new_source.append(line)
    return new_source

# 2. Update Single Editing Requests
new_requests_code = [
    "###### TODO: Use your knowledge. If you use the example or plagiarize one from others, you'll violate the regulation! ######\n",
    "requests = [\n",
    "    {\n",
    "        \"prompt\": \"{} is located in\",\n",
    "        \"subject\": \"The Eiffel Tower\",\n",
    "        \"target_new\": {\n",
    "            \"str\": \"Rome\"\n",
    "        },\n",
    "        \"target_true\": {\n",
    "            \"str\": \"Paris\"\n",
    "        },\n",
    "    }\n",
    "]\n",
    "\n",
    "generation_prompts = [\n",
    "    \"The Eiffel Tower is located in\", # Original Prompt\n",
    "    \"The most famous landmark in Paris is The Eiffel Tower. It is located in\", # Paraphrase Prompt\n",
    "    \"The Louvre is located in\", # Neighborhood Prompt\n",
    "    \"Rome is the location of\", # Reversion Prompt\n",
    "    \"After visiting The Eiffel Tower, you can travel to the nearby city of\" # Portability Prompt\n",
    "]"
]

# 3. Switch to ROME and Add MEMIT comments
def switch_to_rome(source_list):
    new_source = []
    for line in source_list:
        if 'RewritingParamsClass, apply_method, hparam = FTHyperParams, apply_ft_to_model, ft_hparam' in line:
            if not line.strip().startswith('#'):
                 new_source.append('# ' + line.lstrip())
            else:
                 new_source.append(line)
        elif '#RewritingParamsClass, apply_method, hparam = ROMEHyperParams, apply_rome_to_model, rome_hparam' in line:
             new_source.append(line.replace('#RewritingParamsClass', 'RewritingParamsClass'))
        elif '# RewritingParamsClass, apply_method, hparam = ROMEHyperParams, apply_rome_to_model, rome_hparam' in line:
             new_source.append(line.replace('# RewritingParamsClass', 'RewritingParamsClass'))
        else:
            new_source.append(line)
    
    # Add MEMIT suggestion if not present
    if not any('MEMITHyperParams' in line for line in new_source):
        new_source.append('\n')
        new_source.append('# For MEMIT (Multiple Editing):\n')
        new_source.append('# from memit.memit_main import apply_memit_to_model\n')
        new_source.append('# from memit.memit_hparams import MEMITHyperParams\n')
        new_source.append('# RewritingParamsClass, apply_method, hparam = MEMITHyperParams, apply_memit_to_model, MEMITHyperParams.from_json("./hparams/MEMIT/gpt2-xl.json")\n')
        
    return new_source

# 4. Enable all 80 examples for Multiple Editing
def enable_all_examples(source_list):
    new_source = []
    for line in source_list:
        if 'requests = json.load(file)[0:10]' in line and not line.strip().startswith('#'):
            new_source.append('    # requests = json.load(file)[0:10]\n')
        elif '# requests = json.load(file)' in line:
            new_source.append('    requests = json.load(file)\n')
        else:
            new_source.append(line)
    return new_source

# 5. Add Imports
def add_memit_imports(source_list):
    new_source = []
    has_memit = False
    for line in source_list:
        new_source.append(line)
        if 'import memit' in line:
            has_memit = True
    
    if has_memit and not any('apply_memit_to_model' in line for line in new_source):
        new_source.append('from memit.memit_main import apply_memit_to_model\n')
        new_source.append('from memit.memit_hparams import MEMITHyperParams\n')
            
    return new_source

# Iterate cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        source_text = ''.join(source)
        
        # Check for apply_rome_to_model definition
        if 'def apply_rome_to_model' in source_text:
            cell['source'] = fix_apply_rome_source(source)
        
        # Check for imports
        if 'import memit' in source_text:
             cell['source'] = add_memit_imports(source)

        # Check for requests definition (Single Editing) - identifying by unique string
        if 'prompt": "{} was the founder of"' in source_text:
            cell['source'] = new_requests_code

        # Check for method switching
        if 'RewritingParamsClass, apply_method, hparam = FTHyperParams' in source_text:
             cell['source'] = switch_to_rome(source)

        # Check for Multiple Editing data loading
        if 'requests = json.load(file)' in source_text and '0:10' in source_text:
            cell['source'] = enable_all_examples(source)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Notebook modified successfully.")
