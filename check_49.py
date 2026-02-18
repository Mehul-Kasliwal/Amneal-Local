import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pprint import pprint as pp


class ManufacturingProcessTimeValidator:
    """
    Check 49: Verify manufacturing process start and end times.
    ---
    Description:
    1. Find BMR pages with Manufacturing Process section (16.0 MANUFACTURING PROCESS)
    2. Extract process records with Date, Start Time, End Time columns
    3. Verify that:
       - End Time is AFTER Start Time for the same process
       - End Time of a process is BEFORE Start Time of the next process
    4. Handle date changes (processes spanning midnight)
    ---
    Type: Python-based
    ---
    Attachment Required: NO
    ---
    Author: Mehul Kasliwal (06 Feb 2026)
    """

    # ===================== CONSTANTS =====================

    SECTION_NAME = "Manufacturing Process"
    CHECK_NAME = "manufacturing_process_time_verify"

    # Column name synonyms for manufacturing process tables
    START_TIME_SYNONYMS = [
        "Start Time",
        "Start time",
        "start time",
        "Start",
    ]

    END_TIME_SYNONYMS = [
        "End Time",
        "End time",
        "end time",
        "End",
    ]

    DATE_SYNONYMS = [
        "Date",
        "date",
        "DATE",
    ]

    # Keywords to identify manufacturing process pages
    MANUFACTURING_PROCESS_KEYWORDS = [
        "MANUFACTURING PROCESS",
        "Manufacturing Process",
        "16.0 MANUFACTURING PROCESS",
        "MANUFACTURING STEP",
        "PRE-STARTUP ACTIVITY",
    ]

    # Keywords to identify dispensing pre-check pages
    # (e.g. "8.0 DISPENSING OF MATERIALS", room DP checks, "Dispensing of RM/ Excipients")
    DISPENSING_PRECHECKS_KEYWORDS = [
        "DISPENSING OF MATERIALS",
        "Dispensing of Materials",
        "DP of dispensing room",
        "Dispensing of RM/ Excipients",
        "Dispensing of RM/Excipients",
        "8.0 DISPENSING OF MATERIALS",
    ]

    # Keywords to identify raw material dispensing pages
    DISPENSING_KEYWORDS = [
        "RAW MATERIAL DISPENSING",
        "Raw Material Dispensing",
        "10.0 RAW MATERIAL DISPENSING",
        "Dispensing Start Date and Time",
        "Dispensing End Date and Time",
    ]

    # Combined date-time formats for dispensing pages (e.g. "08-APR-25, 11:31")
    DATETIME_COMBINED_FORMATS = [
        "%d-%b-%y, %H:%M",       # 08-Apr-25, 11:31
        "%d-%B-%y, %H:%M",       # 08-April-25, 11:31
        "%d-%b-%Y, %H:%M",       # 08-Apr-2025, 11:31
        "%d-%B-%Y, %H:%M",       # 08-April-2025, 11:31
        "%d/%m/%y, %H:%M",       # 08/04/25, 11:31
        "%d/%m/%Y, %H:%M",       # 08/04/2025, 11:31
        "%d-%m-%y, %H:%M",       # 08-04-25, 11:31
        "%d-%m-%Y, %H:%M",       # 08-04-2025, 11:31
    ]

    # Values considered as blank or NA
    BLANK_VALUES = {"", "na", "na.", "n/a", "n/a.", "n.a", "n.a.", "*", "-", "nil", "none"}

    # Common date formats found in BMR documents
    DATE_FORMATS = [
        "%d-%b-%y",          # 09-Apr-25
        "%d-%b-%Y",          # 09-Apr-2025
        "%d-%B-%y",          # 09-April-25
        "%d-%B-%Y",          # 09-April-2025
        "%d/%m/%y",          # 09/04/25
        "%d/%m/%Y",          # 09/04/2025
        "%d-%m-%y",          # 09-04-25
        "%d-%m-%Y",          # 09-04-2025
    ]

    # ===================== HELPER METHODS =====================

    def is_blank_or_na(self, v: Any) -> bool:
        """Check if a value is blank or represents N/A."""
        if v is None:
            return True
        if not isinstance(v, str):
            return False
        return v.strip().lower() in self.BLANK_VALUES

    def normalize_time(self, time_str: str) -> Optional[str]:
        """
        Normalize time string to HH:MM format.
        Handles various OCR errors and formats.
        E.g., '04:02.' -> '04:02', '65:09' -> None (invalid), '09!:22' -> '09:22'
        Also handles multiple times like '04:52 04:58' -> returns first valid time
        """
        if not isinstance(time_str, str) or self.is_blank_or_na(time_str):
            return None
        
        # Clean up OCR artifacts
        time_str = time_str.strip()
        time_str = time_str.replace('!', ':').replace('.', '')
        time_str = re.sub(r'\s+', ' ', time_str)
        
        # Handle multiple times (e.g., "04:52 04:58") - extract first valid time
        time_parts = time_str.split(' ')
        
        for part in time_parts:
            # Try to match HH:MM pattern
            match = re.match(r'^(\d{1,2}):(\d{2})$', part.strip())
            if match:
                hour, minute = int(match.group(1)), int(match.group(2))
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return f"{hour:02d}:{minute:02d}"
        
        return None

    def normalize_datetime_combined(self, dt_str: str) -> Optional[datetime]:
        """
        Normalize a combined date-time string like '08-APR-25, 11:31' into a datetime object.
        Handles various OCR artifacts.
        """
        if not isinstance(dt_str, str) or self.is_blank_or_na(dt_str):
            return None
        
        # Clean up OCR artifacts
        dt_str = dt_str.strip()
        dt_str = re.sub(r'-\s+', '-', dt_str)
        dt_str = re.sub(r'\s+-', '-', dt_str)
        dt_str = re.sub(r',\s+', ', ', dt_str)  # normalize comma spacing
        dt_str = re.sub(r'\s+', ' ', dt_str)
        
        for fmt in self.DATETIME_COMBINED_FORMATS:
            try:
                parsed = datetime.strptime(dt_str, fmt)
                # Adjust 2-digit year
                if parsed.year < 100:
                    if parsed.year < 50:
                        parsed = parsed.replace(year=parsed.year + 2000)
                    else:
                        parsed = parsed.replace(year=parsed.year + 1900)
                return parsed
            except ValueError:
                continue
        
        return None

    def normalize_date(self, date_str: str) -> Optional[datetime]:
        """
        Normalize date string to datetime object.
        Handles various formats and OCR errors.
        """
        if not isinstance(date_str, str) or self.is_blank_or_na(date_str):
            return None
        
        # Clean up OCR artifacts
        date_str = date_str.strip()
        # Handle common OCR errors like "09-Apr- 25" -> "09-Apr-25"
        date_str = re.sub(r'-\s+', '-', date_str)
        date_str = re.sub(r'\s+-', '-', date_str)
        # Handle "25-10-apr" -> might be OCR error for date
        
        for fmt in self.DATE_FORMATS:
            try:
                parsed = datetime.strptime(date_str, fmt)
                # Adjust year if it's 2-digit and before 50, assume 2000s
                if parsed.year < 100:
                    if parsed.year < 50:
                        parsed = parsed.replace(year=parsed.year + 2000)
                    else:
                        parsed = parsed.replace(year=parsed.year + 1900)
                return parsed
            except ValueError:
                continue
        
        return None

    def parse_datetime(self, date_str: str, time_str: str) -> Optional[datetime]:
        """
        Combine date and time strings into a datetime object.
        """
        date = self.normalize_date(date_str)
        time = self.normalize_time(time_str)
        
        if date is None or time is None:
            return None
        
        hour, minute = map(int, time.split(':'))
        return date.replace(hour=hour, minute=minute)

    def find_col(self, records: List[Dict], synonyms: List[str]) -> Optional[str]:
        """Find which column name variant exists in the records."""
        for syn in synonyms:
            if any(isinstance(r, dict) and syn in r for r in records):
                return syn
        return None

    def iter_dict_rows(self, records: List) -> List[Dict]:
        """Flatten nested record structures, yielding individual dict rows."""
        result = []
        for r in records:
            if isinstance(r, dict):
                result.append(r)
            elif isinstance(r, list):
                result.extend(self.iter_dict_rows(r))
        return result

    # ===================== PAGE DETECTION =====================

    def is_manufacturing_process_page(self, page_data: Dict) -> bool:
        """Check if a page contains manufacturing process data."""
        # Check rules_or_instructions
        rules = page_data.get("rules_or_instructions", [])
        if rules:
            for rule in rules:
                if rule and isinstance(rule, str):
                    for keyword in self.MANUFACTURING_PROCESS_KEYWORDS:
                        if keyword.lower() in rule.lower():
                            return True
        
        # Check markdown_page
        markdown = page_data.get("markdown_page", "")
        if markdown:
            for keyword in self.MANUFACTURING_PROCESS_KEYWORDS:
                if keyword.lower() in markdown.lower():
                    return True
        
        return False

    def has_time_columns(self, records: List[Dict]) -> bool:
        """Check if records have start time and end time columns."""
        flat = self.iter_dict_rows(records)
        if not flat:
            return False
        
        has_start = self.find_col(flat, self.START_TIME_SYNONYMS) is not None
        has_end = self.find_col(flat, self.END_TIME_SYNONYMS) is not None
        
        return has_start and has_end

    # ===================== RECORD EXTRACTION =====================

    def extract_process_records(self, page_data: Dict) -> List[Dict]:
        """Extract manufacturing process records from a page."""
        all_records = []
        records = page_data.get("records", [])
        
        if not records:
            return all_records
        
        flat_records = self.iter_dict_rows(records)
        return flat_records

    def get_process_times(self, records: List[Dict]) -> List[Dict]:
        """
        Extract process times from records with Date, Start Time, End Time.
        Returns list of dicts with parsed datetime values.
        """
        process_times = []
        
        flat_records = self.iter_dict_rows(records)
        if not flat_records:
            return process_times
        
        # Find column names
        start_col = self.find_col(flat_records, self.START_TIME_SYNONYMS)
        end_col = self.find_col(flat_records, self.END_TIME_SYNONYMS)
        date_col = self.find_col(flat_records, self.DATE_SYNONYMS)
        
        if not start_col or not end_col:
            return process_times
        
        for record in flat_records:
            if not isinstance(record, dict):
                continue
            
            start_time_raw = record.get(start_col, "")
            end_time_raw = record.get(end_col, "")
            date_raw = record.get(date_col, "") if date_col else ""
            
            # Skip if both times are blank
            if self.is_blank_or_na(start_time_raw) and self.is_blank_or_na(end_time_raw):
                continue
            
            # Get manufacturing instruction for context
            instruction = ""
            for key in ["Manufacturing instruction", "Manufacturing Instruction", 
                       "Manufacturing instruction", "Pre-start up activity"]:
                if key in record:
                    instruction = record.get(key, "")[:100]
                    break
            
            # Normalize date and times
            date_normalized = self.normalize_date(str(date_raw))
            start_time_normalized = self.normalize_time(str(start_time_raw))
            end_time_normalized = self.normalize_time(str(end_time_raw))
            
            # Format date for output (YYYY-MM-DD format or None)
            date_normalized_str = date_normalized.strftime("%Y-%m-%d") if date_normalized else None
            
            process_times.append({
                "date_raw": date_raw,
                "date_normalized": date_normalized_str,
                "start_time_raw": start_time_raw,
                "end_time_raw": end_time_raw,
                "start_time_normalized": start_time_normalized,
                "end_time_normalized": end_time_normalized,
                "instruction": instruction,
                "original_record": record,
                # Store datetime objects for validation
                "_date_obj": date_normalized,
            })
        
        return process_times

    # ===================== VALIDATION =====================

    def _create_datetime(self, date_obj: Optional[datetime], time_str: Optional[str]) -> Optional[datetime]:
        """
        Create a datetime object from a date and time string.
        Returns None if date or time is missing/invalid.
        """
        if date_obj is None or time_str is None:
            return None
        
        try:
            hour, minute = map(int, time_str.split(':'))
            return date_obj.replace(hour=hour, minute=minute)
        except (ValueError, AttributeError):
            return None

    def validate_process_times(self, process_times: List[Dict], page_no: str) -> List[Dict]:
        """
        Validate process times:
        1. Check if date can be normalized (raise anomaly if not)
        2. End time should be after start time for same process
        3. End time of current process should be before start time of next process
        Uses full datetime comparisons to handle processes spanning different days.
        """
        anomalies = []
        
        for i, process in enumerate(process_times):
            start_time = process.get("start_time_normalized")
            end_time = process.get("end_time_normalized")
            date_raw = process.get("date_raw", "")
            date_obj = process.get("_date_obj")
            date_normalized = process.get("date_normalized")
            instruction = process.get("instruction", "Unknown")[:50]
            
            # Validation 0: Check if date could be normalized (only if date_raw is not blank)
            if date_raw and not self.is_blank_or_na(date_raw) and date_normalized is None:
                anomalies.append({
                    "parameter": f"Date Parsing — Step {i+1}",
                    "issue": f"Unable to normalize date: '{date_raw}'",
                    "date_raw": date_raw,
                    "instruction": instruction
                })
            
            # Create full datetime objects for comparison
            start_datetime = self._create_datetime(date_obj, start_time)
            end_datetime = self._create_datetime(date_obj, end_time)
            
            # Validation 1: End time should be after start time
            if start_time and end_time:
                if start_datetime and end_datetime:
                    # Full datetime comparison (same day assumed for same process)
                    if end_datetime < start_datetime:
                        # Check if it might be a midnight crossing (end is next day)
                        end_next_day = end_datetime + timedelta(days=1)
                        if (end_next_day - start_datetime).total_seconds() / 3600 <= 12:
                            # Valid midnight crossing (process took less than 12 hours)
                            pass
                        else:
                            anomalies.append({
                                "parameter": f"Process Time Sequence — Step {i+1}",
                                "issue": f"End time ({end_time}) is before start time ({start_time}) for the same process",
                                "date": date_normalized,
                                "start_time": start_time,
                                "end_time": end_time,
                                "instruction": instruction
                            })
                else:
                    # Fall back to time-only comparison if date not available
                    start_minutes = int(start_time.split(':')[0]) * 60 + int(start_time.split(':')[1])
                    end_minutes = int(end_time.split(':')[0]) * 60 + int(end_time.split(':')[1])
                    
                    if end_minutes < start_minutes:
                        time_diff = start_minutes - end_minutes
                        if time_diff > 720:  # More than 12 hours difference - likely midnight crossing
                            pass
                        else:
                            anomalies.append({
                                "parameter": f"Process Time Sequence — Step {i+1}",
                                "issue": f"End time ({end_time}) is before start time ({start_time}) for the same process",
                                "date": date_normalized,
                                "start_time": start_time,
                                "end_time": end_time,
                                "instruction": instruction
                            })
            
            # Validation 2: Check sequence with next process
            if i < len(process_times) - 1:
                next_process = process_times[i + 1]
                next_start_time = next_process.get("start_time_normalized")
                next_date_obj = next_process.get("_date_obj")
                next_date_normalized = next_process.get("date_normalized")
                next_instruction = next_process.get("instruction", "Unknown")[:50]
                
                if end_time and next_start_time:
                    # Create datetime for next process start
                    next_start_datetime = self._create_datetime(next_date_obj, next_start_time)
                    
                    if end_datetime and next_start_datetime:
                        # Full datetime comparison considering dates
                        if end_datetime > next_start_datetime:
                            anomalies.append({
                                "parameter": f"Process Sequence — Steps {i+1} to {i+2}",
                                "issue": f"End time of process ({date_normalized} {end_time}) is after start time of next process ({next_date_normalized} {next_start_time})",
                                "current_date": date_normalized,
                                "current_end_time": end_time,
                                "next_date": next_date_normalized,
                                "next_start_time": next_start_time,
                                "current_instruction": instruction,
                                "next_instruction": next_instruction
                            })
                    else:
                        # Fall back to time-only comparison
                        end_minutes = int(end_time.split(':')[0]) * 60 + int(end_time.split(':')[1])
                        next_start_minutes = int(next_start_time.split(':')[0]) * 60 + int(next_start_time.split(':')[1])
                        
                        if end_minutes > next_start_minutes:
                            time_diff = end_minutes - next_start_minutes
                            if time_diff > 720:  # More than 12 hours - likely midnight crossing
                                pass
                            else:
                                anomalies.append({
                                    "parameter": f"Process Sequence — Steps {i+1} to {i+2}",
                                    "issue": f"End time of process ({end_time}) is after start time of next process ({next_start_time})",
                                    "current_end_time": end_time,
                                    "next_start_time": next_start_time,
                                    "current_instruction": instruction,
                                    "next_instruction": next_instruction
                                })
        
        return anomalies

    # ===================== DISPENSING VALIDATION =====================

    def is_dispensing_prechecks_page(self, page_data: Dict) -> bool:
        """Check if a page is a dispensing pre-checks page (e.g. '8.0 DISPENSING OF MATERIALS')."""
        # Check rules_or_instructions
        rules = page_data.get("rules_or_instructions", [])
        if rules:
            for rule in rules:
                if rule and isinstance(rule, str):
                    for keyword in self.DISPENSING_PRECHECKS_KEYWORDS:
                        if keyword.lower() in rule.lower():
                            return True

        # Check markdown_page
        markdown = page_data.get("markdown_page", "")
        if markdown:
            for keyword in self.DISPENSING_PRECHECKS_KEYWORDS:
                if keyword.lower() in markdown.lower():
                    return True

        return False

    def is_dispensing_page(self, page_data: Dict) -> bool:
        """Check if a page is a raw material dispensing page (e.g. '10.0 RAW MATERIAL DISPENSING')."""
        # Check rules_or_instructions
        rules = page_data.get("rules_or_instructions", [])
        if rules:
            for rule in rules:
                if rule and isinstance(rule, str):
                    for keyword in self.DISPENSING_KEYWORDS:
                        if keyword.lower() in rule.lower():
                            return True

        # Check markdown_page
        markdown = page_data.get("markdown_page", "")
        if markdown:
            for keyword in self.DISPENSING_KEYWORDS:
                if keyword.lower() in markdown.lower():
                    return True

        return False

    def _extract_kv_text_blocks(self, page_item: Dict) -> List[Dict]:
        """
        Extract all kv_text_block dicts from a raw page_item.
        Handles both dict and list formats of extracted_kv_text_block.
        """
        kv_blocks = []
        page_content = page_item.get("page_content", [])
        for content_item in page_content:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") == "kv_text_block":
                kv_data = content_item.get("extracted_kv_text_block", {})
                if isinstance(kv_data, list):
                    kv_blocks.extend([k for k in kv_data if isinstance(k, dict)])
                elif isinstance(kv_data, dict):
                    kv_blocks.append(kv_data)
        return kv_blocks

    def extract_dispensing_prechecks_times(
        self, page_item: Dict, page_no: str
    ) -> Tuple[Optional[datetime], List[Dict]]:
        """
        Extract and validate start/end date-time from a dispensing pre-checks page.

        The pre-check page has date, start_time, and end_time as separate fields
        in the kv_text_block (e.g. date='08-APR-25', start_time='25:15', end_time='10:51').

        Returns:
            (end_datetime, anomalies_list)
        """
        anomalies = []
        end_datetime = None

        kv_blocks = self._extract_kv_text_blocks(page_item)
        if not kv_blocks:
            return end_datetime, anomalies

        for kv in kv_blocks:
            date_raw = kv.get("date", "")
            start_time_raw = kv.get("start_time", "")
            end_time_raw = kv.get("end_time", "")

            # Skip if all are blank
            if (self.is_blank_or_na(date_raw)
                    and self.is_blank_or_na(start_time_raw)
                    and self.is_blank_or_na(end_time_raw)):
                continue

            # Normalize date
            date_obj = self.normalize_date(str(date_raw)) if date_raw and not self.is_blank_or_na(date_raw) else None
            if date_raw and not self.is_blank_or_na(date_raw) and date_obj is None:
                anomalies.append({
                    "section": "Dispensing Pre-Checks",
                    "check": "dispensing_prechecks_datetime",
                    "page": page_no,
                    "parameter": "Date Normalization",
                    "issue": f"Unable to normalize date: '{date_raw}'",
                    "date_raw": date_raw,
                })

            # Normalize start time
            start_time_norm = self.normalize_time(str(start_time_raw)) if start_time_raw and not self.is_blank_or_na(start_time_raw) else None
            if start_time_raw and not self.is_blank_or_na(start_time_raw) and start_time_norm is None:
                anomalies.append({
                    "section": "Dispensing Pre-Checks",
                    "check": "dispensing_prechecks_datetime",
                    "page": page_no,
                    "parameter": "Start Time Normalization",
                    "issue": f"Unable to normalize start time: '{start_time_raw}'",
                    "start_time_raw": start_time_raw,
                })

            # Normalize end time
            end_time_norm = self.normalize_time(str(end_time_raw)) if end_time_raw and not self.is_blank_or_na(end_time_raw) else None
            if end_time_raw and not self.is_blank_or_na(end_time_raw) and end_time_norm is None:
                anomalies.append({
                    "section": "Dispensing Pre-Checks",
                    "check": "dispensing_prechecks_datetime",
                    "page": page_no,
                    "parameter": "End Time Normalization",
                    "issue": f"Unable to normalize end time: '{end_time_raw}'",
                    "end_time_raw": end_time_raw,
                })

            # Build full datetimes
            start_dt = self._create_datetime(date_obj, start_time_norm)
            end_dt = self._create_datetime(date_obj, end_time_norm)

            # Validate end > start
            if start_dt and end_dt:
                if end_dt < start_dt:
                    # Check for midnight crossing
                    end_next_day = end_dt + timedelta(days=1)
                    if (end_next_day - start_dt).total_seconds() / 3600 <= 12:
                        # Valid midnight crossing
                        end_dt = end_next_day
                    else:
                        anomalies.append({
                            "section": "Dispensing Pre-Checks",
                            "check": "dispensing_prechecks_datetime",
                            "page": page_no,
                            "parameter": "Time Sequence",
                            "issue": f"End time ({end_time_raw}) is before start time ({start_time_raw})",
                            "date": date_raw,
                            "start_time": start_time_raw,
                            "end_time": end_time_raw,
                        })
            elif end_time_norm and not start_dt:
                # We can still build end_dt for cross-validation even without valid start
                end_dt = self._create_datetime(date_obj, end_time_norm)

            if end_dt is not None:
                end_datetime = end_dt

        return end_datetime, anomalies

    def extract_dispensing_times(
        self, page_item: Dict, page_no: str
    ) -> Tuple[Optional[datetime], Optional[datetime], List[Dict]]:
        """
        Extract and validate start/end date-time from a raw material dispensing page.

        Dispensing pages use combined format: start_time='08-APR-25, 11:31', end_time='08-APR-25, 14:09'.

        Returns:
            (start_datetime, end_datetime, anomalies_list)
        """
        anomalies = []
        start_datetime = None
        end_datetime = None

        kv_blocks = self._extract_kv_text_blocks(page_item)
        if not kv_blocks:
            return start_datetime, end_datetime, anomalies

        for kv in kv_blocks:
            start_time_raw = kv.get("start_time", "")
            end_time_raw = kv.get("end_time", "")

            # Skip if both are blank
            if self.is_blank_or_na(start_time_raw) and self.is_blank_or_na(end_time_raw):
                continue

            # Try combined datetime format first
            start_dt = None
            end_dt = None

            if start_time_raw and not self.is_blank_or_na(start_time_raw):
                start_dt = self.normalize_datetime_combined(str(start_time_raw))
                if start_dt is None:
                    anomalies.append({
                        "section": "Raw Material Dispensing",
                        "check": "dispensing_datetime",
                        "page": page_no,
                        "parameter": "Start Date-Time Normalization",
                        "issue": f"Unable to normalize dispensing start date-time: '{start_time_raw}'",
                        "start_time_raw": start_time_raw,
                    })

            if end_time_raw and not self.is_blank_or_na(end_time_raw):
                end_dt = self.normalize_datetime_combined(str(end_time_raw))
                if end_dt is None:
                    anomalies.append({
                        "section": "Raw Material Dispensing",
                        "check": "dispensing_datetime",
                        "page": page_no,
                        "parameter": "End Date-Time Normalization",
                        "issue": f"Unable to normalize dispensing end date-time: '{end_time_raw}'",
                        "end_time_raw": end_time_raw,
                    })

            # Validate end > start
            if start_dt and end_dt:
                if end_dt < start_dt:
                    end_next_day = end_dt + timedelta(days=1)
                    if (end_next_day - start_dt).total_seconds() / 3600 <= 12:
                        end_dt = end_next_day
                    else:
                        anomalies.append({
                            "section": "Raw Material Dispensing",
                            "check": "dispensing_datetime",
                            "page": page_no,
                            "parameter": "Time Sequence",
                            "issue": f"End date-time ({end_time_raw}) is before start date-time ({start_time_raw})",
                            "start_time": start_time_raw,
                            "end_time": end_time_raw,
                        })

            if start_dt is not None:
                start_datetime = start_dt
            if end_dt is not None:
                end_datetime = end_dt

        return start_datetime, end_datetime, anomalies

    def validate_dispensing_process(
        self, filled_master_json: List[Dict]
    ) -> List[Dict]:
        """
        Validate the dispensing process date-time sequence:
        1. Find dispensing pre-check pages -> validate start/end, store end_datetime
        2. Find dispensing pages -> validate start/end, compare start to pre-check end

        Returns list of debug/anomaly dicts.
        """
        bmr_data = self._preprocess_filled_master_json(filled_master_json)
        return_debug = []

        prechecks_end_time = None
        prechecks_page_no = None

        # --- Pass 1: Dispensing pre-check pages ---
        for page_item in filled_master_json:
            page_no = str(page_item.get("page", ""))
            if not page_no:
                continue

            page_data = bmr_data.get(page_no, {})
            if not self.is_dispensing_prechecks_page(page_data):
                continue

            # Also skip pages that match dispensing (raw material) keywords
            # to avoid double-matching (the pre-check page heading is "DISPENSING OF MATERIALS"
            # but the actual dispensing page is "RAW MATERIAL DISPENSING")
            if self.is_dispensing_page(page_data):
                continue

            end_dt, anomalies = self.extract_dispensing_prechecks_times(page_item, page_no)

            if end_dt is not None:
                prechecks_end_time = end_dt
                prechecks_page_no = page_no

            return_debug.append({
                "page_no": page_no,
                "section_name": "Dispensing Pre-Checks",
                "check_name": "dispensing_prechecks_datetime",
                "anomaly_status": 1 if anomalies else 0,
                "prechecks_end_time": prechecks_end_time.strftime("%Y-%m-%d %H:%M") if prechecks_end_time else None,
                "anomalies": anomalies,
            })

        # --- Pass 2: Dispensing pages ---
        for page_item in filled_master_json:
            page_no = str(page_item.get("page", ""))
            if not page_no:
                continue

            page_data = bmr_data.get(page_no, {})
            if not self.is_dispensing_page(page_data):
                continue

            start_dt, end_dt, anomalies = self.extract_dispensing_times(page_item, page_no)

            # Cross-validate: dispensing start should be >= pre-checks end
            if start_dt and prechecks_end_time:
                if start_dt < prechecks_end_time:
                    anomalies.append({
                        "section": "Raw Material Dispensing",
                        "check": "dispensing_datetime",
                        "page": page_no,
                        "parameter": "Dispensing vs Pre-Checks Sequence",
                        "issue": (
                            f"Dispensing start ({start_dt.strftime('%Y-%m-%d %H:%M')}) "
                            f"is before dispensing pre-checks end "
                            f"({prechecks_end_time.strftime('%Y-%m-%d %H:%M')}) on page {prechecks_page_no}"
                        ),
                        "dispensing_start": start_dt.strftime("%Y-%m-%d %H:%M"),
                        "prechecks_end": prechecks_end_time.strftime("%Y-%m-%d %H:%M"),
                        "prechecks_page": prechecks_page_no,
                    })

            return_debug.append({
                "page_no": page_no,
                "section_name": "Raw Material Dispensing",
                "check_name": "dispensing_datetime",
                "anomaly_status": 1 if anomalies else 0,
                "dispensing_start": start_dt.strftime("%Y-%m-%d %H:%M") if start_dt else None,
                "dispensing_end": end_dt.strftime("%Y-%m-%d %H:%M") if end_dt else None,
                "anomalies": anomalies,
            })

        return return_debug

    # ===================== ENTRY POINT =====================

    def _preprocess_filled_master_json(self, filled_master_json: List[Dict]) -> Dict[str, Dict]:
        """
        Convert raw filled_master_json list format to the dict format used internally.
        
        Input format (raw filled_master_json from production):
            [{"page": 1, "page_content": [...], "markdown_page": "...", "additional_keys": [...]}, ...]
        
        Output format:
            {"1": {"records": [...], "markdown_page": "...", "rules_or_instructions": [...]}, ...}
        """
        bmr_data = {}
        
        for page_item in filled_master_json:
            page_no = str(page_item.get("page", ""))
            if not page_no:
                continue
                
            page_content = page_item.get("page_content", [])
            additional_keys = page_item.get("additional_keys", [])
            
            # Extract records from page_content
            # page_content is a list of items like {"type": "table", "table_json": {...}, "config": {...}}
            all_records = []
            rules_or_instructions = []
            
            for content_item in page_content:
                if not isinstance(content_item, dict):
                    continue
                    
                table_json = content_item.get("table_json", {})
                content_type = content_item.get("type", "")
                
                # Check if this table_json has 'records' key (manufacturing process tables)
                if isinstance(table_json, dict) and "records" in table_json:
                    records = table_json.get("records", [])
                    if isinstance(records, list):
                        all_records.extend(records)
                
                # Check for rules_or_instructions in table_json
                if isinstance(table_json, dict) and "rules_or_instructions" in table_json:
                    roi = table_json.get("rules_or_instructions", [])
                    if isinstance(roi, list):
                        rules_or_instructions.extend(roi)
                
                # Check extracted_kv_text_block for rules_or_instructions
                if content_type == "kv_text_block":
                    kv_data = content_item.get("extracted_kv_text_block", {})
                    if isinstance(kv_data, list):
                        for kv in kv_data:
                            if isinstance(kv, dict) and "rules_or_instructions" in kv:
                                roi = kv.get("rules_or_instructions", [])
                                if isinstance(roi, list):
                                    rules_or_instructions.extend(roi)
                    elif isinstance(kv_data, dict) and "rules_or_instructions" in kv_data:
                        roi = kv_data.get("rules_or_instructions", [])
                        if isinstance(roi, list):
                            rules_or_instructions.extend(roi)
            
            # Also check additional_keys for rules_or_instructions
            if isinstance(additional_keys, list):
                for ak in additional_keys:
                    if isinstance(ak, dict) and "rules_or_instructions" in ak:
                        roi = ak.get("rules_or_instructions", [])
                        if isinstance(roi, list):
                            rules_or_instructions.extend(roi)
            elif isinstance(additional_keys, dict) and "rules_or_instructions" in additional_keys:
                roi = additional_keys.get("rules_or_instructions", [])
                if isinstance(roi, list):
                    rules_or_instructions.extend(roi)
            
            # Build page data dict
            page_data = {
                "records": all_records,
                "markdown_page": page_item.get("markdown_page", ""),
                "rules_or_instructions": rules_or_instructions,
            }
            bmr_data[page_no] = page_data
        
        return bmr_data

    def run_validation(
        self,
        filled_master_json: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Main entry point for manufacturing process time validation.

        Args:
            filled_master_json: Raw filled_master_json list from production
                Format: [{"page": 1, "page_content": [...], "markdown_page": "...", "additional_keys": [...]}, ...]

        Returns:
            List of result dicts with detailed debug information including process_times and anomalies
        """
        # Preprocess the raw filled_master_json to internal format
        bmr_data = self._preprocess_filled_master_json(filled_master_json)
        
        results = []

        # ---- Dispensing Process Validation ----
        dispensing_debug = self.validate_dispensing_process(filled_master_json)
        return_debug = []  # Detailed debug information

        # Collect all manufacturing process records across pages for cross-page validation
        all_process_times = []
        page_process_map = {}  # Map page_no to process times on that page

        for page_no, page_data in bmr_data.items():
            try:
                # Check if this is a dispensing page and skip if so (handled separately)
                if self.is_dispensing_prechecks_page(page_data) or self.is_dispensing_page(page_data):
                    continue

                # Check if this is a manufacturing process page
                if not self.is_manufacturing_process_page(page_data):
                    results.append({
                        "page": int(page_no),
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0
                    })
                    continue

                # Extract process records
                records = self.extract_process_records(page_data)
                if not records:
                    results.append({
                        "page": int(page_no),
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0
                    })
                    continue

                # Check if records have time columns
                if not self.has_time_columns(records):
                    results.append({
                        "page": int(page_no),
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0
                    })
                    continue

                # Get process times
                process_times = self.get_process_times(records)
                if not process_times:
                    results.append({
                        "page": int(page_no),
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0
                    })
                    continue

                # Store for cross-page validation
                page_process_map[page_no] = process_times
                for pt in process_times:
                    pt["page_no"] = page_no
                all_process_times.extend(process_times)

                # Validate within page
                page_anomalies = self.validate_process_times(process_times, page_no)

                # Build process times summary for debug (without original_record to keep it clean)
                process_times_summary = []
                for pt in process_times:
                    process_times_summary.append({
                        "date_raw": pt.get("date_raw"),
                        "date_normalized": pt.get("date_normalized"),
                        "start_time_raw": pt.get("start_time_raw"),
                        "end_time_raw": pt.get("end_time_raw"),
                        "start_time_normalized": pt.get("start_time_normalized"),
                        "end_time_normalized": pt.get("end_time_normalized"),
                        "instruction": pt.get("instruction", "")[:100]
                    })

                if page_anomalies:
                    return_debug.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 1,
                        "process_times": process_times_summary,
                        "anomalies": page_anomalies
                    })
                    results.append({
                        "page": int(page_no),
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 1
                    })
                else:
                    return_debug.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0,
                        "process_times": process_times_summary,
                        "anomalies": []
                    })
                    results.append({
                        "page": int(page_no),
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0
                    })

            except Exception as e:
                return_debug.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0,
                    "error": str(e),
                    "anomalies": []
                })
                results.append({
                    "page": int(page_no),
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0
                })

        # Cross-page validation: Check if last process of page N ends before first process of page N+1
        sorted_pages = sorted(page_process_map.keys(), key=int)
        for i, page_no in enumerate(sorted_pages[:-1]):
            next_page_no = sorted_pages[i + 1]
            
            current_page_times = page_process_map.get(page_no, [])
            next_page_times = page_process_map.get(next_page_no, [])
            
            if current_page_times and next_page_times:
                last_process = current_page_times[-1]
                first_next_process = next_page_times[0]
                
                last_end_time = last_process.get("end_time_normalized")
                first_start_time = first_next_process.get("start_time_normalized")
                
                if last_end_time and first_start_time:
                    last_end_minutes = int(last_end_time.split(':')[0]) * 60 + int(last_end_time.split(':')[1])
                    first_start_minutes = int(first_start_time.split(':')[0]) * 60 + int(first_start_time.split(':')[1])
                    
                    if last_end_minutes > first_start_minutes:
                        time_diff = last_end_minutes - first_start_minutes
                        if time_diff <= 720:  # Not a midnight crossing
                            # Update the result for next_page_no to show anomaly
                            for result in results:
                                if result["page"] == int(next_page_no):
                                    result["anomaly_status"] = 1
                                    break

        # Append dispensing validation results
        return_debug.extend(dispensing_debug)

        # Add dispensing entries to results (production output)
        for disp_entry in dispensing_debug:
            results.append({
                "page": int(disp_entry.get("page_no", 0)),
                "section_name": disp_entry.get("section_name", ""),
                "check_name": disp_entry.get("check_name", ""),
                "anomaly_status": disp_entry.get("anomaly_status", 0),
            })

        # Sort all results by page number for clean output
        return_debug.sort(key=lambda x: int(x.get("page_no", 0)))
        results.sort(key=lambda x: x.get("page", 0))

        return results


# =====================================================
# ---------------- MAIN RUNNER ------------------------
# =====================================================
if __name__ == "__main__":
    import os

    # File paths
    bmr_filepath = '/home/softsensor/Desktop/Amneal/complete_data_61 1.json'

    # Load data
    with open(bmr_filepath, "r", encoding="utf-8") as f:
        complete_data = json.load(f)
    
    # Extract filled_master_json from steps object
    filled_master_json = complete_data.get("steps", {}).get("filled_master_json", [])
    
    if not filled_master_json:
        print("Error: Could not find 'filled_master_json' in steps")
        exit(1)
    
    print(f"Loaded {len(filled_master_json)} pages from filled_master_json")

    # Run validation - pass raw filled_master_json directly
    # Preprocessing is done inside run_validation
    validator = ManufacturingProcessTimeValidator()
    results = validator.run_validation(filled_master_json)

    # Output as JSON
    print(json.dumps(results, indent=2))

    
