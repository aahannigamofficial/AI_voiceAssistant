"""
Intent recognition module.

Parses user commands and extracts structured intents.
Example: "open YouTube" → {action: "open_website", target: "youtube"}
"""

import re
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Intent:
    """Represents a recognized intent."""
    action: str  # e.g., "open_website", "launch_app", "search_web"
    target: Optional[str] = None  # e.g., "youtube", "notepad"
    query: Optional[str] = None  # e.g., search query or text to type
    confidence: float = 0.0  # 0.0 to 1.0
    
    def __str__(self):
        if self.query:
            return f"{self.action}(target='{self.target}', query='{self.query}')"
        elif self.target:
            return f"{self.action}(target='{self.target}')"
        else:
            return f"{self.action}()"


class IntentRecognizer:
    """Recognizes intents from natural language commands."""
    
    # Mapping of common names to standardized targets
    WEBSITE_ALIASES = {
        "youtube": "https://youtube.com",
        "google": "https://google.com",
        "gmail": "https://gmail.com",
        "github": "https://github.com",
        "reddit": "https://reddit.com",
        "twitter": "https://twitter.com",
        "linkedin": "https://linkedin.com",
        "facebook": "https://facebook.com",
    }
    
    APP_ALIASES = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "settings": "ms-settings:",
        "file explorer": "explorer.exe",
        "vscode": "code",
        "vs code": "code",
        "visual studio code": "code",
        "chrome": "chrome",
        "edge": "msedge",
        "firefox": "firefox",
        "word": "winword",
        "excel": "excel",
        "powerpoint": "powerpnt",
        "paint": "mspaint",
    }
    
    def __init__(self):
        """Initialize the intent recognizer."""
        self.patterns = [
            # Open website: "open youtube", "go to google", "visit reddit"
            (r"(?:open|go to|visit|browse|check)\s+([a-z\s]+?)(?:\s+(?:website|page))?$", 
             self._handle_website),
            
            # Launch application: "launch notepad", "open vs code", "start calculator"
            (r"(?:launch|open|start|run)\s+([a-z\s]+?)(?:\s+(?:application|app))?$", 
             self._handle_app),
            
            # Search web: "search google for machine learning", "google xgboost"
            (r"(?:search|google|find)\s+(?:(?:google\s+)?for\s+)?(.+)$", 
             self._handle_search),
            
            # Type text: "type hello world"
            (r"type\s+(.+)$", 
             self._handle_type),
            
            # Press key: "press enter", "press escape"
            (r"press\s+([a-z]+)$", 
             self._handle_key),
            
            # Click: "click" (will require screen context later)
            (r"click", 
             self._handle_click),
        ]
    
    def recognize(self, text: str) -> Optional[Intent]:
        """
        Recognize intent from text.
        
        Args:
            text: The user's spoken command (already transcribed).
        
        Returns:
            Intent object with action and parameters, or None if not recognized.
        """
        text = text.lower().strip()
        
        # Remove wake word if present
        text = re.sub(r"^hey\s+(?:mycroft|jarvis|rhasspy|assistant)\s+", "", text)
        text = text.strip()
        
        if not text:
            return None
        
        # Try each pattern
        for pattern, handler in self.patterns:
            match = re.match(pattern, text)
            if match:
                return handler(match)
        
        return None
    
    def _handle_website(self, match) -> Intent:
        """Handle website opening intent."""
        target = match.group(1).strip().lower()

        # "open" is accepted for both websites and applications.
        if target in self.APP_ALIASES:
            return Intent(
                action="launch_app",
                target=self.APP_ALIASES[target],
                confidence=0.9
            )

        target = target.replace(" ", "")
        
        # Look up URL
        url = self.WEBSITE_ALIASES.get(target, f"https://{target}.com")
        
        return Intent(
            action="open_website",
            target=url,
            confidence=0.9
        )
    
    def _handle_app(self, match) -> Intent:
        """Handle application launch intent."""
        target = match.group(1).strip().lower()
        
        # Look up app
        app_path = self.APP_ALIASES.get(target, target)
        
        return Intent(
            action="launch_app",
            target=app_path,
            confidence=0.9
        )
    
    def _handle_search(self, match) -> Intent:
        """Handle web search intent."""
        query = match.group(1).strip()
        
        return Intent(
            action="search_web",
            target="https://google.com",
            query=query,
            confidence=0.8
        )
    
    def _handle_type(self, match) -> Intent:
        """Handle typing text intent."""
        text = match.group(1).strip()
        
        return Intent(
            action="type_text",
            query=text,
            confidence=0.85
        )
    
    def _handle_key(self, match) -> Intent:
        """Handle key press intent."""
        key = match.group(1).strip().lower()
        
        return Intent(
            action="press_key",
            target=key,
            confidence=0.9
        )
    
    def _handle_click(self, match) -> Intent:
        """Handle click intent (will need screen context)."""
        return Intent(
            action="click",
            confidence=0.5
        )