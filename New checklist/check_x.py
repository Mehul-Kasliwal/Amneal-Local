REQUIRED_ATTACHMENT_KEYS = {
        "Attachment No",
        "Title / Details",
        "No. of pages",
    }

#pass entire page_content as key to this function
def recursive_search(self, obj: Any) -> bool:
        if isinstance(obj, dict):
            if "records" in obj and isinstance(obj["records"], list):
                for record in obj["records"]:
                    if (
                        isinstance(record, dict)
                        and self.REQUIRED_ATTACHMENT_KEYS.issubset(record.keys())
                    ):
                        return True
            return any(self.recursive_search(v) for v in obj.values())

        if isinstance(obj, list):
            return any(self.recursive_search(i) for i in obj)

        return False

#It has to be run like below
for idx, page in enumerate(document_pages, start=1):
    page_content = page.get("page_content", [])
    if self.recursive_search(page_content):
