import time
import os
import requests
import json
from flask import Flask, request, send_file
import io
import tempfile
from langchain.document_loaders.unstructured import UnstructuredFileLoader
import cv2
import re
from typing import List, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import atexit
logger = logging.getLogger(__name__)
from PIL import Image
import numpy as np

PDF_OCR_THRESHOLD = (0.6, 0.6)
import tqdm

app = Flask(__name__)
temp_file_path = None
ans_list = []

prompts = [
    "该保险的险种类型-主附险是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的险种类型-长短险是？（保险期间长于1年的为长险，反之为短险。在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的设计类型是？（可多选，其中类型包括预定收益型，分红型，传统型，万能型，投连型，其他。在回答时只需回复答案即可，没有答案则回复无。）"
    "该保险的险种类型是？（其中类型包含定期寿险, 重大疾病保险, 税延养老年金, 终身寿险, 医疗保险, 寿险, 两全险, 养老年金:年金险+保险期间长(终身), 失能收入损失保险, 护理保险, 投资型财险, 商业养老金, 专属商业养老, 子女教育年金:年金险+保险期间短(30年以下), 车险, 财产保障险, 责任险, 信用险, 意外险, 其他保险。且在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的有效保额增长率是？（例如：本合同第一个保单年度的有效保险金额等于本合同的基本保险金额；自第二个保单年度起，本合同当年度有效保险金额等于本合同上一个保单年度的有效保险金额×（1+3%）。则文中的有效保额增长率为3%。且在回答时只需回复答案即可，没有则回答无）",
    "该保险的投保年龄范围是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的保费要求是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的保费要求是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的保险期间是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的交费期间是？（在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的交费方式是？（例如：保险费的交费方式分为年交和月交。文中的交费方式是年交，月交。在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的等待期是？（回复格式：xx天 ，在回答时只需回复答案即可，没有答案则回复无）",
    "该保险的犹豫期是多少天？（回复格式：xx天 ，不需要有多余的解释，如果根据已知信息无法回答，则回复“我无法回答您的问题”。例如：“本合同生效后，自您收到保险单的次日零时起，您享有 30 日的犹豫期，以便您在此期间阅读本合同。”中的犹豫期为30日）",
    "该保险的宽限期是？（回复格式：xx天 ，在回答时只需回复答案即可，没有答案则回复无）",
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


def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    try:
        from rapidocr_paddle import RapidOCR
        ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    return ocr


class RapidOCRPDFLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def rotate_img(img, angle):
            '''
            img   --image
            angle --rotation angle
            return--rotated img
            '''

            h, w = img.shape[:2]
            rotate_center = (w / 2, h / 2)
            # 获取旋转矩阵
            # 参数1为旋转中心点;
            # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
            # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
            M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
            # 计算图像新边界
            new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
            new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
            # 调整旋转矩阵以考虑平移
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2

            rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
            return rotated_img

        def pdf2text(file):
            import fitz  # pyMuPDF里面的fitz包，不要与pip install fitz混淆
            import numpy as np
            ocr = get_ocr()
            doc = fitz.Document(stream=pdf_bytes, filetype='pdf')
            resp = ""

            b_unit = tqdm.tqdm(total=doc.page_count, desc="RapidOCRPDFLoader context page index: 0")
            for i, page in enumerate(doc):
                b_unit.set_description("RapidOCRPDFLoader context page index: {}".format(i))
                b_unit.refresh()
                text = page.get_text("")
                resp += text + "\n"

                img_list = page.get_image_info(xrefs=True)
                for img in img_list:
                    if xref := img.get("xref"):
                        bbox = img["bbox"]
                        # 检查图片尺寸是否超过设定的阈值
                        if ((bbox[2] - bbox[0]) / (page.rect.width) < PDF_OCR_THRESHOLD[0]
                                or (bbox[3] - bbox[1]) / (page.rect.height) < PDF_OCR_THRESHOLD[1]):
                            continue
                        pix = fitz.Pixmap(doc, xref)
                        samples = pix.samples
                        if int(page.rotation) != 0:  # 如果Page有旋转角度，则旋转图片
                            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                            tmp_img = Image.fromarray(img_array);
                            ori_img = cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR)
                            rot_img = rotate_img(img=ori_img, angle=360 - page.rotation)
                            img_array = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
                        else:
                            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)

                        result, _ = ocr(img_array)
                        if result:
                            ocr_result = [line[1] for line in result]
                            resp += "\n".join(ocr_result)

                # 更新进度
                b_unit.update(1)
            return resp

        text = pdf2text(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(text=text, **self.unstructured_kwargs)


def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
            # splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s"
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip() != ""]


@app.route('/upload_and_chat', methods=['POST'])
# 第一步：上传文件并获取id
def upload_temp_docs():
    # 检查请求中是否有文件
    import tempfile
    global temp_file_path
    global pdf_bytes

    if 'file' not in request.files:
        return 'No file part in the request'

    # 获取上传的文件
    uploaded_file = request.files['file']
    pdf_bytes = uploaded_file.read()

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


        loader = RapidOCRPDFLoader(files)
        ls = loader.load()
        ls = str(ls[0]).replace("page_content=", "").replace("'", "")
        # print(ls,type(ls))
        ls = ls.split("\n\n")
        # print(ls, type(ls))
        # print(ls, type(ls))
        # ls = ['"""'+ls+'"""',]
        text_splitter = ChineseRecursiveTextSplitter(
            keep_separator=True,
            is_separator_regex=True,
            chunk_size=50,
            chunk_overlap=0
        )

        found_value = False
        for inum, text in enumerate(ls):
            # print(inum)
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                if not found_value:
                    # print("****第一行***",chunk)
                    break
                # print(chunk)

        chunk = re.sub(r'\s+([a-zA-Z])\s+', r'\1', chunk)
        chunk1 = re.split("\\n\\n", chunk)
        chunk2 = re.split(" ", chunk)
        # print(chunk)

        for c in chunk1:
            print(c + "\\n")
            matches = re.findall(r'\b\d{4}\b', c)
            if c.endswith("条款") and c != "请扫描以查询验证条款":
                tiaokuan = c

            else:
                for d in chunk2:
                    if d.endswith("条款") and d != "请扫描以查询验证条款":
                        tiaokuan = d

            if matches:
                for match in matches:
                    niandu = match


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
                # print(type(response1.text))

                if response1.status_code == 200:
                    result = response1.text
                    return result.split(":")[-1].replace("}", "".replace('"', ''))
                else:
                    print(f"chat failed with status code {response1.status_code}: {response1.text}")
                    return None

            output = io.StringIO()
            output.write("条款名称 {}\\n".format(tiaokuan))
            output.write("保险年度 {}\\n".format(niandu))

            for prompt in prompts:
                answer = chat(prompt)
                if answer is not None:
                    output.write(
                        prompt.split("（")[0].replace("该保险的", "").replace("根据文中的title告诉我，", "").replace(
                            "该保险", "").replace("是？",
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
    if temp_file_path:
        try:
            os.unlink(temp_file_path)
        except PermissionError:
            print("File is still in use, will try to delete later.")
        except Exception as e:
            print(f"Failed to delete temp file: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7863, debug=True)
