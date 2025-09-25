import os
import sys
import copy
from typing import List, Iterator
import cv2
from PIL import Image
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredFileLoader
import tqdm
import numpy as np
import pymupdf
import re
from wired_table_rec import WiredTableRecognition
from .base_loader import BaseLoader  # 导入基础接口

# PDF OCR 控制参数
PDF_OCR_THRESHOLD = (0.6, 0.6)


class CustomizedOcrPdfLoader(UnstructuredFileLoader, BaseLoader):
    """支持OCR的PDF加载器"""
    
    table_enhance: List = None

    def __init__(self, file_path: str):
        UnstructuredFileLoader.__init__(self, file_path=file_path)
        BaseLoader.__init__(self, file_path=file_path)

    def _is_paragraph_end(self, text):
        if text.strip()[-1] in ["。", "？", ".", "?"]:
            return True
        return False

    def _get_elements(self) -> List:
        def rotate_img(img, angle):
            h, w = img.shape[:2]
            rotate_center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
            new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
            new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2
            return cv2.warpAffine(img, M, (new_w, new_h))

        def clip_text_and_table(page):
            resp = ""
            resp_list = []
            tabs = page.find_tables()
            last_tab_bbox_y1 = page.rect.y0
            
            for tab in tabs:
                tab_bbox = pymupdf.Rect(tab.bbox)
                top_bbox = page.rect
                top_bbox.y0 = last_tab_bbox_y1
                top_bbox.y1 = tab_bbox.y0
                text = page.get_text("", clip=top_bbox)
                
                if text:
                    for sub_text in text.split("\n"):
                        if not sub_text.strip() or sub_text.strip().isdigit():
                            continue
                        if self._is_paragraph_end(sub_text):
                            resp_list.append(resp + sub_text)
                            resp = ""
                        else:
                            resp += sub_text.strip()
                
                tab_md_text = tab.to_pandas().to_markdown().replace(" ", "").replace("\n", "")
                if resp:
                    resp_list.append(resp + tab_md_text)
                    resp = ""
                else:
                    resp_list.append(tab_md_text)
                
                last_tab_bbox_y1 = tab_bbox.y1

            btm_bbox = page.rect
            btm_bbox.y0 = last_tab_bbox_y1
            text = page.get_text("", clip=btm_bbox)
            
            if text:
                for sub_text in text.split("\n"):
                    if not sub_text.strip() or sub_text.strip().isdigit():
                        continue
                    if self._is_paragraph_end(sub_text):
                        resp_list.append(resp + sub_text)
                        resp = ""
                    else:
                        resp += sub_text.strip()

            return resp_list

        def pdf2text(filepath):
            table_rec = WiredTableRecognition()
            from rag.module.loader.ocr import get_rapid_ocr
            rapid_ocr = get_rapid_ocr()
            doc = pymupdf.open(filepath)
            resp = ""
            resp_list = []
            tab_resp_list = []

            b_unit = tqdm.tqdm(total=doc.page_count, desc="PDF加载进度")
            for i, page in enumerate(doc):
                b_unit.set_description(f"正在加载第{i+1}页")
                text = page.get_text("")
                
                if text:
                    for sub_text in text.split("\n"):
                        if not sub_text.strip() or sub_text.strip().isdigit():
                            continue
                        if self._is_paragraph_end(sub_text):
                            resp_list.append(resp + sub_text)
                            resp = ""
                        else:
                            resp += sub_text.strip()

                img_list = page.get_image_info(xrefs=True)
                for img in img_list:
                    if xref := img.get("xref"):
                        bbox = img["bbox"]
                        if ((bbox[2] - bbox[0]) / page.rect.width < PDF_OCR_THRESHOLD[0] or 
                            (bbox[3] - bbox[1]) / page.rect.height < PDF_OCR_THRESHOLD[1]):
                            continue
                            
                        pix = pymupdf.Pixmap(doc, xref)
                        if int(page.rotation) != 0:
                            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                            tmp_img = Image.fromarray(img_array)
                            ori_img = cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR)
                            rot_img = rotate_img(img=ori_img, angle=360 - page.rotation)
                            img_array = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
                        else:
                            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)

                        rapid_ocr_result, _ = rapid_ocr(img_array)
                        if rapid_ocr_result:
                            rapid_ocr_result = [line[1] for line in rapid_ocr_result]
                            for text in rapid_ocr_result:
                                if text.strip().isdigit():
                                    continue
                                if self._is_paragraph_end(text):
                                    resp_list.append(resp + text)
                                    resp = ""
                                else:
                                    resp += text.strip()

                        table_str, elapse = table_rec(img_array)
                        if table_str:
                            table_ele_index = []
                            for index, t in enumerate(rapid_ocr_result):
                                if t in table_str:
                                    table_ele_index.append(index)
                            index = min(table_ele_index) if table_ele_index else 0
                            table_name = rapid_ocr_result[:index][-1] if rapid_ocr_result[:index] else ""
                            tab_resp_list.append([table_name, table_str])

                tabs = page.find_tables()
                for tab in tabs:
                    header = tab.header
                    external = header.external
                    table_name = ""
                    
                    if external:
                        table_name = "".join(header.names)
                    else:
                        tab_bbox = pymupdf.Rect(tab.bbox)
                        top_bbox = page.rect
                        top_bbox.y1 = tab_bbox.y0
                        top_text = page.get_text("", clip=top_bbox)
                        for t in top_text.split("\n")[::-1]:
                            if t.strip():
                                table_name = t
                                break
                        if not table_name and tab_resp_list:
                            table_name = tab_resp_list[-1][0]
                            
                    tab_text = tab.to_pandas().to_markdown(index=False)
                    tab_text = re.sub(r'-{2,}', '---', tab_text)
                    tab_text = re.sub(r' {2,}', ' ', tab_text)
                    tab_resp_list.append([table_name, tab_text])

                b_unit.update(1)
            
            if tab_resp_list:
                self.table_enhance = [table_info[0] + "\n" + table_info[1] for table_info in tab_resp_list]
            if resp:
                resp_list.append(resp)
            return "\n".join(resp_list)

        text = pdf2text(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(text=text, **self.unstructured_kwargs)

    def load(self) -> List[Document]:
        """实现基础接口的load方法"""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """实现基础接口的lazy_load方法"""
        elements = self._get_elements()
        metadata = self._get_metadata()
        
        if not self.table_enhance:
            yield Document(page_content="\n\n".join([str(el) for el in elements]), metadata=metadata)
        else:
            yield Document(page_content="\n\n".join([str(el) for el in elements]), metadata=metadata)
            for table in self.table_enhance:
                yield Document(page_content=table, metadata=copy.deepcopy(metadata))

    @classmethod
    def supported_extensions(cls) -> List[str]:
        """返回支持的文件扩展名"""
        return [".pdf"]

    