from .law_parser import LawParser
from .decree_parser import DecreeParser
from .qcvn_parser import QcvnParser
from .circular_parser import CircularParser

class DocumentRouter:
    """Factory to determine and instantiate the appropriate structural parser based on document metadata and content."""
    
    @staticmethod
    def get_parser(filename: str, md_text: str, doc_name: str = ""):
        text_lower = md_text[:2000].lower()
        fname_lower = filename.lower()
        dname_lower = doc_name.lower()
        
        if any(keyword in fname_lower or keyword in dname_lower for keyword in ["qcvn", "quy chuẩn"]):
            return QcvnParser()
        if "quy chuẩn" in text_lower:
            return QcvnParser()

        if any(keyword in fname_lower or keyword in dname_lower for keyword in ["qh", "luật"]):
            return LawParser()
        if "luật" in text_lower[:200]:
            return LawParser()

        if any(keyword in fname_lower or keyword in dname_lower for keyword in ["tt-bgtvt", "bgtvt", "thông tư"]):
            return CircularParser()
        if "thông tư" in text_lower[:500]:
            return CircularParser()

        if any(keyword in fname_lower or keyword in dname_lower for keyword in ["nd-cp", "nđ-cp", "nghị định"]):
            return DecreeParser()
            
        return DecreeParser()
