"""
Computer interaction executor.

Safely executes allowed actions on the computer.
All actions are allowlisted - no arbitrary code execution.
"""

import subprocess
import webbrowser
import time
from typing import Optional
import pyautogui


class ComputerExecutor:
    """Executes safe, allowlisted computer actions."""
    
    def __init__(self, enable_execution: bool = True):
        """
        Initialize executor.
        
        Args:
            enable_execution: If False, only logs what would be executed (dry run).
        """
        self.enable_execution = enable_execution
    
    def open_website(self, url: str, delay: float = 1.0) -> bool:
        """
        Open a website in the default browser.
        
        Args:
            url: The website URL (e.g., "https://youtube.com").
            delay: Delay before opening (in seconds).
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            if not url.startswith("http"):
                url = f"https://{url}"
            
            print(f"Opening website: {url}")
            
            if self.enable_execution:
                time.sleep(delay)
                webbrowser.open(url)
            
            return True
        except Exception as e:
            print(f"Failed to open website: {e}")
            return False
    
    def launch_app(self, app_path: str, delay: float = 0.5) -> bool:
        """
        Launch an application.
        
        Args:
            app_path: Application path or name (e.g., "notepad.exe", "code").
            delay: Delay before launching (in seconds).
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            print(f"Launching application: {app_path}")
            
            if self.enable_execution:
                time.sleep(delay)
                subprocess.Popen(app_path)
            
            return True
        except Exception as e:
            print(f"Failed to launch application: {e}")
            return False
    
    def search_web(self, query: str, engine: str = "google") -> bool:
        """
        Search the web.
        
        Args:
            query: Search query.
            engine: Search engine ("google", "bing", etc.).
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Build search URL
            if engine.lower() == "google":
                url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            elif engine.lower() == "bing":
                url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
            else:
                url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            
            print(f"Searching {engine} for: {query}")
            
            if self.enable_execution:
                webbrowser.open(url)
            
            return True
        except Exception as e:
            print(f"Failed to search: {e}")
            return False
    
    def type_text(self, text: str, delay: float = 0.1) -> bool:
        """
        Type text using keyboard.
        
        Args:
            text: The text to type.
            delay: Delay between keystrokes (in seconds).
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            print(f"Typing: {text}")
            
            if self.enable_execution:
                # Small delay to let user focus on the input field
                time.sleep(0.5)
                pyautogui.write(text, interval=delay)
            
            return True
        except Exception as e:
            print(f"Failed to type text: {e}")
            return False
    
    def press_key(self, key: str) -> bool:
        """
        Press a keyboard key.
        
        Args:
            key: Key name (e.g., "enter", "escape", "backspace").
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            print(f"Pressing key: {key}")
            
            if self.enable_execution:
                pyautogui.press(key)
            
            return True
        except Exception as e:
            print(f"Failed to press key: {e}")
            return False
    
    def click(self, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """
        Click at a position.
        
        Args:
            x: X coordinate. If None, uses current position.
            y: Y coordinate. If None, uses current position.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            if x is not None and y is not None:
                print(f"Clicking at ({x}, {y})")
                if self.enable_execution:
                    pyautogui.click(x, y)
            else:
                print("Clicking at current position")
                if self.enable_execution:
                    pyautogui.click()
            
            return True
        except Exception as e:
            print(f"Failed to click: {e}")
            return False
    
    def scroll(self, direction: str = "down", amount: int = 3) -> bool:
        """
        Scroll the mouse wheel.
        
        Args:
            direction: "up" or "down".
            amount: Number of clicks to scroll.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            scroll_amount = amount if direction.lower() == "down" else -amount
            print(f"Scrolling {direction} ({amount} clicks)")
            
            if self.enable_execution:
                pyautogui.scroll(scroll_amount)
            
            return True
        except Exception as e:
            print(f"Failed to scroll: {e}")
            return False
    
    def close_application(self, app_name: str) -> bool:
        """
        Close an application.
        
        Args:
            app_name: Application name (e.g., "notepad", "chrome").
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            print(f"Closing application: {app_name}")
            
            if self.enable_execution:
                # Windows-specific
                subprocess.run(f"taskkill /IM {app_name}.exe", shell=True)
            
            return True
        except Exception as e:
            print(f"Failed to close application: {e}")
            return False
    
    def execute_intent(self, intent) -> bool:
        """
        Execute an intent.
        
        Args:
            intent: Intent object with action and parameters.
        
        Returns:
            True if successful, False otherwise.
        """
        if intent.action == "open_website":
            return self.open_website(intent.target)
        elif intent.action == "launch_app":
            return self.launch_app(intent.target)
        elif intent.action == "search_web":
            return self.search_web(intent.query)
        elif intent.action == "type_text":
            return self.type_text(intent.query)
        elif intent.action == "press_key":
            return self.press_key(intent.target)
        elif intent.action == "click":
            return self.click()
        elif intent.action == "scroll":
            return self.scroll()
        else:
            print(f"Unknown action: {intent.action}")
            return False