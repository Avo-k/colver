#!/usr/bin/env python3
"""GitHub Webhook Server for Colver Auto-Deploy."""

import hashlib
import hmac
import logging
import os
import subprocess
import sys
from pathlib import Path

from flask import Flask, request, jsonify

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

WEBHOOK_SECRET = os.environ.get('WEBHOOK_SECRET', '')
SCRIPT_DIR = Path(__file__).parent.absolute()
DEPLOY_SCRIPT = SCRIPT_DIR / 'deploy.sh'
TARGET_BRANCH = 'master'


def verify_signature(payload: bytes, signature: str) -> bool:
    if not WEBHOOK_SECRET:
        logger.error("WEBHOOK_SECRET not configured")
        return False

    # Try SHA-256 first, fall back to SHA-1
    if signature.startswith('sha256='):
        expected = 'sha256=' + hmac.new(
            WEBHOOK_SECRET.encode(), payload, hashlib.sha256
        ).hexdigest()
    elif signature.startswith('sha1='):
        expected = 'sha1=' + hmac.new(
            WEBHOOK_SECRET.encode(), payload, hashlib.sha1
        ).hexdigest()
    else:
        return False

    return hmac.compare_digest(expected, signature)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'colver-webhook'})


@app.route('/webhook', methods=['POST'])
def webhook():
    signature = request.headers.get('X-Hub-Signature-256') or request.headers.get('X-Hub-Signature', '')
    payload = request.get_data()

    if not verify_signature(payload, signature):
        logger.warning("Invalid webhook signature")
        return jsonify({'error': 'Invalid signature'}), 401

    event = request.headers.get('X-GitHub-Event', '')
    if event == 'ping':
        logger.info("Received ping from GitHub")
        return jsonify({'message': 'pong'})
    if event != 'push':
        return jsonify({'message': f'Event {event} ignored'})

    data = request.get_json()
    branch = data.get('ref', '').replace('refs/heads/', '')
    if branch != TARGET_BRANCH:
        logger.info(f"Ignoring push to {branch}")
        return jsonify({'message': f'Push to {branch} ignored'})

    pusher = data.get('pusher', {}).get('name', 'unknown')
    commits = data.get('commits', [])
    logger.info(f"Push to {branch} from {pusher} ({len(commits)} commits)")

    try:
        result = subprocess.run(
            [str(DEPLOY_SCRIPT)],
            cwd=str(SCRIPT_DIR),
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            logger.info("Deployment successful")
            return jsonify({'message': 'Deployed', 'commits': len(commits)})
        else:
            logger.error(f"Deploy failed: {result.stderr}")
            return jsonify({'error': result.stderr}), 500
    except subprocess.TimeoutExpired:
        logger.error("Deploy timed out")
        return jsonify({'error': 'Timeout'}), 500


if __name__ == '__main__':
    if not WEBHOOK_SECRET:
        logger.warning("WEBHOOK_SECRET not set!")
    logger.info("Starting colver webhook server on port 9002")
    app.run(host='0.0.0.0', port=9002)
