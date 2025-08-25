get_ngrok_url() {
  sleep 2
  URL=$(curl --silent http://127.0.0.1:4040/api/tunnels | \
        grep -o '"public_url":"https://[^"]*' | \
        grep -o 'https://[^"]*')
  
  echo $URL
}


if [ "$1" == "--ngrok" ]; then
  ### NGROK MODE ###
  
  echo "Starting services in background and launching ngrok..."

  docker-compose up --build -d

  # Check if the 'ngrok' command is available in the system's PATH
  if ! command -v ngrok &> /dev/null; then
    echo "------------------------------------------------------------"
    echo "ERROR: 'ngrok' command not found."
    echo "Please download ngrok from https://ngrok.com/download"
    echo "and ensure the executable is in your system's PATH."
    echo "------------------------------------------------------------"
    # Stop the background containers before exiting
    docker-compose down
    exit 1
  fi

  echo "Starting ngrok tunnel for port 3000..."
  ngrok http 3000 --log=ngrok.log &

  # Get the public URL from ngrok's API
  NGROK_URL=$(get_ngrok_url)

  # Check if we successfully got the URL
  if [ -z "$NGROK_URL" ]; then
    echo "------------------------------------------------------------"
    echo "ERROR: Could not retrieve ngrok URL."
    echo "Is the ngrok agent running correctly? Check 'ngrok.log' for details."
    echo "You can also visit http://127.0.0.1:4040 in your browser."
    echo "------------------------------------------------------------"
    docker-compose down
    exit 1
  fi

  echo "------------------------------------------------------------"
  echo "✅ Your application is running!"
  echo ""
  echo "➡️ Your Public URL is: $NGROK_URL"
  echo ""
  echo "Open this public URL in your LAPTOP's browser."
  echo "Then, scan the QR code that appears with your PHONE."
  echo "------------------------------------------------------------"
  echo "Press Ctrl+C in this terminal to shut everything down."
  echo "------------------------------------------------------------"

  trap "echo; echo 'Shutting down ngrok and Docker containers...'; docker-compose down; killall ngrok &> /dev/null; exit" INT
  
  while true; do sleep 1; done

else
  ### DEFAULT MODE ###
  
  echo "Starting WebRTC VLM Demo in foreground..."
  docker-compose up --build
fi