/**
 * WebRTC Connector for Decart Lucy 2 / Fal.ai Real-Time VTON
 * Manages low-latency video streaming to cloud generative models
 */

export class DecartWebRTCClient {
  constructor() {
    this.peerConnection = null;
    this.dataChannel = null;
    this.isConnected = false;
    this.onRemoteTrack = null;
    this.onStatusChange = null;
    this.apiKey = localStorage.getItem("DECART_API_KEY") || "";
  }

  setApiKey(key) {
    this.apiKey = key;
    localStorage.setItem("DECART_API_KEY", key);
  }

  getApiKey() {
    return this.apiKey;
  }

  /**
   * Connect to Decart Lucy 2 VTON Realtime Endpoint
   * @param {MediaStream} localStream - User webcam stream
   */
  async connect(localStream) {
    if (!this.apiKey) {
      throw new Error("Decart API Key is required for Cloud Generative Mode. Please enter your API key in Settings.");
    }

    this.notifyStatus("Connecting to Decart Lucy-2 Realtime WebRTC...");

    try {
      // Create WebRTC Peer Connection
      const configuration = {
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
      };
      this.peerConnection = new RTCPeerConnection(configuration);

      // Add local video track to send to Decart
      localStream.getTracks().forEach((track) => {
        this.peerConnection.addTrack(track, localStream);
      });

      // Handle incoming generative video track from Lucy 2
      this.peerConnection.ontrack = (event) => {
        console.log("Received remote generative stream from Decart Lucy 2:", event);
        if (this.onRemoteTrack) {
          this.onRemoteTrack(event.streams[0]);
        }
      };

      // Create DataChannel for low-latency prompt / ornament switching
      this.dataChannel = this.peerConnection.createDataChannel("decart-control", {
        ordered: true
      });

      this.dataChannel.onopen = () => {
        this.isConnected = true;
        this.notifyStatus("Connected (Lucy 2 VTON Active)");
      };

      this.dataChannel.onclose = () => {
        this.isConnected = false;
        this.notifyStatus("Disconnected from Decart");
      };

      // Create SDP Offer
      const offer = await this.peerConnection.createOffer();
      await this.peerConnection.setLocalDescription(offer);

      // In production with Decart/Fal:
      // Send offer to Decart signaling endpoint via HTTP POST:
      // const response = await fetch("https://api.decart.ai/v1/realtime/connect", {
      //   method: "POST",
      //   headers: {
      //     "Authorization": `Bearer ${this.apiKey}`,
      //     "Content-Type": "application/json"
      //   },
      //   body: JSON.stringify({ sdp: offer.sdp, model: "decart/lucy2-vton/realtime" })
      // });
      // const answer = await response.json();
      // await this.peerConnection.setRemoteDescription(new RTCSessionDescription(answer));

      this.notifyStatus("WebRTC Session Initialized (Awaiting Model Stream)");
      return true;
    } catch (error) {
      console.error("Decart WebRTC Error:", error);
      this.notifyStatus(`Connection Error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Update the active ornament reference image and prompt in real time
   */
  async updateOrnament(ornament) {
    if (!this.isConnected || !this.dataChannel) {
      console.log(`[Decart Client] Switched ornament to: ${ornament.name} (Simulated WebRTC payload sent)`);
      return;
    }

    const payload = JSON.stringify({
      type: "update_garment",
      image_url: ornament.image,
      category: ornament.category,
      prompt: `photorealistic ${ornament.name}, intricate ${ornament.metal} with ${ornament.gems}, natural skin contact, sharp reflections`,
      strength: 0.8
    });

    this.dataChannel.send(payload);
  }

  notifyStatus(status) {
    if (this.onStatusChange) {
      this.onStatusChange(status);
    }
  }

  disconnect() {
    if (this.peerConnection) {
      this.peerConnection.close();
      this.peerConnection = null;
    }
    this.isConnected = false;
    this.notifyStatus("Offline");
  }
}
