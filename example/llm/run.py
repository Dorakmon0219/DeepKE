import os
import json
import hydra
from hydra import utils
import logging
from easyinstruct.prompts import IEPrompt
from preprocess import prepare_examples
from time import sleep
logger = logging.getLogger(__name__)
@hydra.main( config_path="conf", config_name="config")
def main(cfg):
    results = []
    cfg.cwd = utils.get_original_cwd()


    if not cfg.api_key:
        raise ValueError("Need an API Key.")
    if cfg.engine not in ["text-davinci-003", "text-davinci-002", "text-curie-001", "text-babbage-001", "text-ada-001", "gpt-3.5-turbo"]:
        raise ValueError("The OpenAI model is not supported now.")

    os.environ['OPENAI_API_KEY'] = cfg.api_key

    ie_prompter = IEPrompt(cfg.task)

    examples = None
    if not cfg.zero_shot:
        examples = prepare_examples(cfg.data_path, cfg.task, cfg.language)
    writer = open(r"C:\Users\86150\Desktop\DeepKE\example\llm\valid1_output.json", 'a+', encoding='utf-8')

    with open(r"C:\Users\86150\Desktop\DeepKE\example\llm\valid1.json",'r',encoding='utf-8') as file:
        i=0
        for f in file:
            i=i+1
            if i > 500:
                break
            item = json.loads(f)
            f_instruction=item["instruction"]
            f_input = item["input"]
            print(item)
            if cfg.task == 're':
                ie_prompter.build_prompt(
                    prompt=f_input,
                    head_entity=cfg.head_entity,
                    head_type=cfg.head_type,
                    tail_entity=cfg.tail_entity,
                    tail_type=cfg.tail_type,
                    language=cfg.language,
                    instruction=cfg.instruction,
                    in_context=not cfg.zero_shot,
                    domain=cfg.domain,
                    labels=cfg.labels,
                    examples=examples
                )
            else:
                ie_prompter.build_prompt(
                    prompt=f_instruction+f_input,
                    language=cfg.language,
                    instruction=cfg.instruction,
                    in_context=not cfg.zero_shot,
                    domain=cfg.domain,
                    labels=cfg.labels,
                    examples=examples
                )
            result = ie_prompter.get_openai_result()
            print(f"gpt response-----------{result}------------")
            item['output'] = result
            results.append(result)
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")
            sleep(21)
    print(f"结果是{results}")
    #logger.info(result)


if __name__ == '__main__':
    main()