import sys
import os
import json
from unittest.mock import MagicMock, patch

# Add the current directory to sys.path
sys.path.append(os.getcwd())

# Create a mock for google.genai.types
mock_types = MagicMock()

def test_format_messages():
    # We must patch before importing or during execution of _format_messages_for_gemini
    # since it imports inside the function.
    
    from agent.vertex_gemini import _format_messages_for_gemini

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help?", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get-weather", "arguments": '{"location": "San Francisco"}'}}
        ]},
        {"role": "tool", "name": "get-weather", "content": '{"temp": 72, "unit": "F"}'}
    ]

    # Setup the mock behavior
    def mock_content(role, parts):
        return {"role": role, "parts": parts}
    
    mock_types.Content.side_effect = mock_content
    mock_types.Part.from_text.side_effect = lambda text: {"text": text}
    mock_types.Part.from_function_call.side_effect = lambda name, args: {"function_call": {"name": name, "args": args}}
    mock_types.Part.from_function_response.side_effect = lambda name, response: {"function_response": {"name": name, "response": response}}

    with patch.dict('sys.modules', {'google.genai': MagicMock(), 'google.genai.types': mock_types}):
        system_instruction, contents = _format_messages_for_gemini(messages)

    print("System Instruction:", system_instruction)
    print("\nContents:")
    for i, content in enumerate(contents):
        role = content['role']
        print(f"Message {i} ({role}):")
        for part in content['parts']:
            print(f"  {part}")

    # Assertions
    assert system_instruction == "You are a helpful assistant.\n\nBe concise."
    assert contents[0]['role'] == "user"
    assert contents[1]['role'] == "model"
    # Assistant message should have text part AND function_call part
    assert len(contents[1]['parts']) == 2
    assert contents[1]['parts'][0]['text'] == "Hi there! How can I help?"
    assert "function_call" in contents[1]['parts'][1]
    assert contents[1]['parts'][1]['function_call']['name'] == "get_weather"
    
    assert contents[2]['role'] == "user"
    assert "function_response" in contents[2]['parts'][0]
    assert contents[2]['parts'][0]['function_response']['name'] == "get_weather"
    assert contents[2]['parts'][0]['function_response']['response'] == {"temp": 72, "unit": "F"}

    print("\nVerification Successful!")

if __name__ == "__main__":
    try:
        test_format_messages()
    except Exception as e:
        print(f"Verification Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
