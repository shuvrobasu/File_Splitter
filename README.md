<img width="1920" height="1128" alt="image" src="https://github.com/user-attachments/assets/08c76a68-e580-4648-9e53-f853e4b34c47" />
A simple tool I desgined to copy paste code in WebUI of AI Chats
Since each Chat tool has a different context length for input and pasting a large amount of text
at one go causes an error, waste of time and effort. Also at times you may not need to share your entire script
but just a part. This script allows you to also split a part (E.g. a Class or method or Function).

You can split code in 3 ways
- By Lines (x lines)
- By Tokens (useful for Chat AI)
- By Parts (x parts)
The ** key difference ** between Lines and Parts is how granular the split is. For e.g a codebase of 10k lines can be split into 5 parts (which may not be equal parts or 2000 lines
per part). In both cases, the script will try to approximate it as best as possible so that a method/function is not split midway into two differnt parts.

This is specially useful when you just need a subset of your code


