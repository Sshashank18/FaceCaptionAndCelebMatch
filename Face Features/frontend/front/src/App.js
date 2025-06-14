import React, { useState } from "react";
import axios from "axios";
import {
  Container,
  Typography,
  Button,
  Box,
  CircularProgress,
  Alert,
  Stack,
} from "@mui/material";
import { styled } from "@mui/system";
import "./App.css"

const Input = styled("input")({
  display: "none",
});

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [caption, setCaption] = useState("");
  const [matchedActor, setMatchedActor] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [actorImageUrl, setActorImageUrl] = useState("");
  const [uploadImageUrl, setUploadImageUrl] = useState("");

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setCaption("");
    setMatchedActor("");
    setError("");
    setUploadImageUrl(URL.createObjectURL(e.target.files[0]))
  };

  const sendImage = async (type) => {
    if (!selectedFile) {
      setError("Please upload an image first!");
      return;
    }

    setLoading(true);
    setError("");
    setCaption("");
    setMatchedActor("");

    try {
      const formData = new FormData();
      formData.append("image", selectedFile);

      const response = await axios.post("http://localhost:3500/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (type === "caption") {
        setCaption(response.data.caption);
      } else if (type === "match") {
        setMatchedActor(
          `Matched Actor: ${response.data.matched_celebrity} (Similarity: ${response.data.similarity_score})`
        );
        setActorImageUrl(response.data.actor_image_url);
      }
    } catch (err) {
      console.error(err);
      setError("Error processing image.");
    }
    setLoading(false);
  };

  return (
    <div className="app">
      <Container maxWidth="sm" sx={{ mt: 5, fontFamily: "Roboto" }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom align="center">
          Celebrity Caption & Matching
        </Typography>

        <Box textAlign="center" mt={3}>
          <label htmlFor="upload-photo">
            <Input
              accept="image/*"
              id="upload-photo"
              type="file"
              onChange={handleFileChange}
            />
            <Button variant="contained" component="span" color="primary">
              Upload Image
            </Button>
          </label>
        </Box>

        <Stack direction="row" spacing={2} justifyContent="center" mt={4}>
          <Button
            variant="contained"
            color="secondary"
            onClick={() => sendImage("caption")}
            disabled={loading}
          >
            Generate Caption
          </Button>
          <Button
            variant="contained"
            color="success"
            onClick={() => sendImage("match")}
            disabled={loading}
          >
            Find Similar Actor
          </Button>
        </Stack>
        {uploadImageUrl && (
          <Box mt={2}>
            <Typography variant="subtitle1">Uploaded Image:</Typography>
            <img
              src={uploadImageUrl}
              alt="Uploaded"
              style={{ maxWidth: "100%", borderRadius: 10 }}
            />
          </Box>
        )}
        <Box mt={4} textAlign="center">
          {loading && <CircularProgress />}
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
          {caption && (
            <Typography variant="subtitle1" mt={2}>
              <strong>Caption:</strong> {caption}
            </Typography>
          )}
          {matchedActor && (
            <>
              <Typography variant="subtitle1" mt={2}>
                {matchedActor}
              </Typography>
              {actorImageUrl && (
                <Box mt={2}>
                  <img
                    src={actorImageUrl}
                    alt="Matched Actor"
                    style={{ width:"300px",height:"250px", borderRadius: 10 }}
                  />
                </Box>
              )}
            </>
          )}
        </Box>
      </Container>
    </div>

  );
}

export default App;
