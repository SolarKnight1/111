import time
import os
import requests
import json
from flask import Flask, request, send_file
import io
import tempfile

app = Flask(__name__)

ans_list = []

prompts = [
    "该保险的合同名称（条款名称）是？（例如：中国人民健康保险股份有限公司人保健康悠越保互联网特定药品费用医疗保险条款 中的合同名称（条款名称）是人保健康悠越保互联网特定药品费用医疗保险，且在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的险种类型（主附险）是?（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的险种类型（长短险）是？（保险期间长于1年的为长险，反之为短险。在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的险种类型是？（其中包含医疗险，重疾险，意外险，定期寿险，定额终身寿，增额终身寿，两全险，年金险，万能险，分红险，投连险，财产险，宠物险，且在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的报备年度是？（一般从文章第一段中提取，例如人保健康[2023]医疗保险 087 号中的报备年度2023年，且在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的有效保额增长率是？（例如：本合同第一个保单年度的有效保险金额等于本合同的基本保险金额；自第二个保单年度起，本合同当年度有效保险金额等于本合同上一个保单年度的有效保险金额×（1+3%）。则文中的有效保额增长率为3%。且在回答时只需回复答案即可，没有则回答无）",
    "该保险的投保年龄范围是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的保费要求是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的保费要求是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的保险期间是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的交费期间是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的交费方式是？（例如：保险费的交费方式分为年交和月交。文中的交费方式是年交，月交。在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的等待期是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的犹豫期是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的宽限期是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的健康告知是？（例如：重要提示：请您如实填写投保资料、如实告知有关情况。<br>根据《中华人民共和国保险法》的规定，您应当真实、完整告知投保流程中的各项信息。保险公司就您及被保险人的有关情况提出询问的，包括但不限于职业状况、财务状况、健康状况、生活方式和习惯、既往承保情况等，您应当如实告知，不得隐瞒或不实告知。否则保险公司有权依据法律规定及本保险合同约定解除保险合同并不承担保险赔偿责任。<br><br>被保险人健康问询如下：<br>1.&nbsp;&nbsp;&nbsp;&nbsp;您是否有危险嗜好或从事危险活动，如赛车、赛马、滑雪、攀岩、蹦极、潜水、跳水、跳伞、拳击、武术、摔跤、探险、特技活动、极限运动、非民用或非商业性飞行或其他高风险活动？<br>2.&nbsp;&nbsp;&nbsp;&nbsp;您是否曾被保险公司解除合同或投保、复效时被拒保、延期、加费、以附加条件承保、提出或已经得到理赔（医疗险一年内理赔不超过3次以及每次理赔金额不超过3000元的情况除外）？<br>3.&nbsp;&nbsp;&nbsp;&nbsp;您是否曾使用止痛药、镇静剂、麻醉剂、迷幻药等成瘾性药物或毒品？您是否曾患有或疑似患有下列疾病、病症或残疾，或者因此而接受就诊、检查或治疗？<br>4.&nbsp;&nbsp;&nbsp;&nbsp;您是否曾患有、或疑似患有下列病症，或因此而就诊、接受检查或治疗？精神疾病、癫痫、脑瘫、脑血管疾病、脑脊髓膜炎、脑炎、脑损伤、昏迷、帕金森氏病、阿耳茨海默氏病、重症肌无力、瘫痪、运动神经元疾病、脊髓灰质炎、癌症、肿瘤、结节、肿物、包块、III度烧烫伤、高血压、心血管及心脏疾病（例如：风湿性心脏病、冠心病、心肌梗塞、心肌炎、心瓣膜疾病、心肌病、心包疾病、心功能不全等）、肺动脉高压、肺纤维化、终末期肺病、呼吸功能不全、各种慢性中毒、性传播疾病、艾滋病或艾滋病病毒携带、慢性肝炎、慢性肝病、肝硬化、慢性肾炎、肾病综合征、肾功能不全、多囊肾、血友病、白血病、再生障碍性贫血、糖尿病、系统性红斑狼疮、类风湿性关节炎、多发性硬化症、器官移植术或造血干细胞移植术、冠状动脉旁路移植术、早产儿体重不足1000克（2周岁及以下儿童适用）、先天性疾病、遗传性疾病，智能障碍，失明、聋哑等听觉、视觉、语言及咀嚼功能障碍，身体部位的缺损、残障、畸形及功能障碍。文中的健康告知为1.&nbsp;&nbsp;&nbsp;&nbsp;您是否有危险嗜好或从事危险活动，如赛车、赛马、滑雪、攀岩、蹦极、潜水、跳水、跳伞、拳击、武术、摔跤、探险、特技活动、极限运动、非民用或非商业性飞行或其他高风险活动？<br>2.&nbsp;&nbsp;&nbsp;&nbsp;您是否曾被保险公司解除合同或投保、复效时被拒保、延期、加费、以附加条件承保、提出或已经得到理赔（医疗险一年内理赔不超过3次以及每次理赔金额不超过3000元的情况除外）？<br>3.&nbsp;&nbsp;&nbsp;&nbsp;您是否曾使用止痛药、镇静剂、麻醉剂、迷幻药等成瘾性药物或毒品？您是否曾患有或疑似患有下列疾病、病症或残疾，或者因此而接受就诊、检查或治疗？<br>4.&nbsp;&nbsp;&nbsp;&nbsp;您是否曾患有、或疑似患有下列病症，或因此而就诊、接受检查或治疗？精神疾病、癫痫、脑瘫、脑血管疾病、脑脊髓膜炎、脑炎、脑损伤、昏迷、帕金森氏病、阿耳茨海默氏病、重症肌无力、瘫痪、运动神经元疾病、脊髓灰质炎、癌症、肿瘤、结节、肿物、包块、III度烧烫伤、高血压、心血管及心脏疾病（例如：风湿性心脏病、冠心病、心肌梗塞、心肌炎、心瓣膜疾病、心肌病、心包疾病、心功能不全等）、肺动脉高压、肺纤维化、终末期肺病、呼吸功能不全、各种慢性中毒、性传播疾病、艾滋病或艾滋病病毒携带、慢性肝炎、慢性肝病、肝硬化、慢性肾炎、肾病综合征、肾功能不全、多囊肾、血友病、白血病、再生障碍性贫血、糖尿病、系统性红斑狼疮、类风湿性关节炎、多发性硬化症、器官移植术或造血干细胞移植术、冠状动脉旁路移植术、早产儿体重不足1000克（2周岁及以下儿童适用）、先天性疾病、遗传性疾病，智能障碍，失明、聋哑等听觉、视觉、语言及咀嚼功能障碍，身体部位的缺损、残障、畸形及功能障碍。在回答时只需回复答案即可，没有答案则回复无）",
    "该保险要求的职业类别是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险要求的体检要求是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险是否保证续保？（在回答时只需回复答案即可，没有则返回无）",
    "该保险的保证续保年龄是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的最大续保年龄是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险停售是否可续保？（在回答时只需回复答案即可，没有则返回无）",
    "该保险的责任免除数量是？（即责任免除下的小标题数量。例如：因下列情形之一导致被保险人身故的，我们不承担给付身故保险金的责任： 1.投保人对被保险人的故意杀害、故意伤害； 2.被保险人故意犯罪或者抗拒依法采取的刑事强制措施； 3.被保险人自伤,或自本合同成立或者合同效力恢复之日起 2 年内自杀（但被保 险人自杀时为无民事行为能力人的除外）； 4.被保险人主动吸食或注射毒品（见 9.4）； 5.被保险人酒后驾驶（见 9.5），无合法有效驾驶证驾驶（见 9.6），或驾驶无有效 行驶证（见 9.7）的机动车（见 9.8）； 6.战争、军事冲突、暴乱或武装叛乱； 7.核爆炸、核辐射或核污染。 发生上述第 1 项情形导致被保险人身故的，本合同终止，我们向被保险人的继 承人退还本合同的现金价值（见 5.1）。 发生上述其他情形导致被保险人身故的，本合同终止，我们向您退还本合同的 现金价值。上文中的责任免除数量为7条。在回答时只需回复答案即可，没有则返回无）",
    "该保险的责任免除描述是？（即责任免除标题下的全文。例如：第八条 责任免除 因下列情形之一导致被保险人身故的，我们不承担给付身故保险金的责任： （1）您对被保险人的故意杀害、故意伤害； （2）被保险人故意犯罪或抗拒依法采取的刑事强制措施； （3）被保险人故意自伤、或自本合同成立或者本合同效力恢复之日起二年内自杀，但被保险人自杀时为无民 事行为能力人的除外； （4）被保险人服用、吸食或注射毒品7 ； （5）被保险人酒后驾驶8 、无合法有效驾驶证驾驶9 ，或驾驶无合法有效行驶证10的机动车11； （6）战争、军事冲突、暴乱或武装叛乱； （7）核爆炸、核辐射或核污染。 发生上述第（1）项情形导致被保险人身故的，本合同终止，我们向被保险人的继承人退还其身故时本合同的 现金价值。在回答时只需回复答案即可，没有则返回无。）",
    "该保险的特色服务时哪些？（类别有就医绿通，健康管理，体检服务，专家门诊，国内二次诊疗，就医陪诊，住院安排，康复护理，机票报销，精准医疗。可多选，且在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的就医绿通是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的健康管理是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的体检服务是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的专家门诊是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的住院安排是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的国内二次诊疗是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的就医陪诊是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的康复护理是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的机票报销是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的精准医疗是？（在回答时只需回复答案即可，没有答案则回复无）"
]

import tempfile


@app.route('/upload_and_chat', methods=['POST'])
# 第一步：上传文件并获取id
def upload_temp_docs():
    # 检查请求中是否有文件
    if 'file' not in request.files:
        return 'No file part in the request'

    # 获取上传的文件
    uploaded_file = request.files['file']

    # 检查文件名是否为空
    if uploaded_file.filename == '':
        return 'No selected file'

    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name

    try:
        # 将上传的文件保存到临时文件中
        uploaded_file.save(temp_file_path)

        url = "http://116.62.199.56:7861/knowledge_base/upload_temp_docs"
        payload = {'data': '{"chunk_size": 256,"chunk_overlap": 128,"zh_title_enhance": True,}'}
        files = [
            ('files', (uploaded_file.filename, open(temp_file_path, 'rb'), 'application/pdf'))
        ]
        headers = {}
        response = requests.request("POST", url, headers=headers, data=payload, files=files)

        # 其余代码保持不变
        if response.status_code == 200:
            result = response.json()
            zsk_id = result.get('data').get('id')

            def chat(prompt):
                url1 = "http://116.62.199.56:7861/chat/file_chat"
                headers1 = {
                    'Content-Type': 'application/json'
                }
                payload1 = {'query': prompt,
                            'knowledge_id': zsk_id,
                            'top_k': 3,
                            'score_threshold': 1.0,
                            'history': [],
                            'stream': True,
                            'model_name': "zhipu-api",
                            'temperature': 0.1,
                            'max_tokens': 8196,
                            'prompt_name': 'default',
                            }
                response1 = requests.request("POST", url1, headers=headers1, json=payload1)
                print(f"Status code: {response1.status_code}")
                print(f"Response text: {response1.text}")
                print(type(response1.text))

                if response1.status_code == 200:
                    result = response1.text
                    return result.split(":")[-1].replace("}", "".replace('"', ''))
                else:
                    print(f"chat failed with status code {response1.status_code}: {response1.text}")
                    return None

            output = io.StringIO()
            for prompt in prompts:
                answer = chat(prompt)
                if answer is not None:
                    output.write(prompt.split("(")[0].replace("该保险的", "").replace("该保险", "").replace("是？",
                                                                                                            "").split("(")[0] + answer + "\n")
                    # print(answer)
                time.sleep(1)
                # print(result.get('data').get('id'))
            # return zsk_id
        else:
            print(f"Upload failed with status code {response.status_code}: {response.text}")
            return None

            # 将 StringIO 中的文本内容读取到变量中
        text_content = output.getvalue()

        # 将文本内容编码为字节流
        byte_content = text_content.encode('utf-8')

        # 将字节流包装到一个 BytesIO 对象中
        byte_stream = io.BytesIO(byte_content)
        # with open("result{}.txt".format(files_path.split('\\')[-1]).replace('.pdf', ''), "w",encoding="utf-8") as f:
        #     f.write(str(ans_list))

        return send_file(byte_stream, as_attachment=True,
                         download_name="result{}.txt".format(uploaded_file.filename.split('\\')[-1]).replace('.pdf',
                                                                                                             ''))

    finally:
        # 确保在发生异常时关闭临时文件
        temp_file.close()


import atexit


# 注册一个函数，在程序退出时删除临时文件
@atexit.register
def cleanup_temp_file():
    global temp_file_path
    if temp_file_path:
        try:
            os.unlink(temp_file_path)
        except PermissionError:
            print("File is still in use, will try to delete later.")
        except Exception as e:
            print(f"Failed to delete temp file: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7863, debug=True)
