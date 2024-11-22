3. Run this script and follow these steps:
When prompted, open the authorization URL in your browser.
Log in and authorize the application.
You'll be redirected to a URL starting with https://localhost/?. Copy this entire URL.
Paste this URL back into your console when prompted.
The script should then be able to exchange the fresh authorization code for an access token.
Remember these important points:
Always use the most recent authorization URL and code.
Authorization codes can only be used once and expire quickly (usually within minutes).
If you encounter this error again, repeat the process to get a new authorization code.
If you continue to face issues, double-check your client ID and client secret, and ensure that the redirect URI in your Google Developers Console matches the one you're using in your code (in this case, it should be https://localhost/).