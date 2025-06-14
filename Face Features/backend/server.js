const express = require('express');
const multer = require('multer');
const axios = require('axios');
const cors = require('cors');
const fs = require('fs');
const FormData = require('form-data');  // THIS one is the node package

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(cors());


app.post('/upload', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image uploaded' });
  }

  try {
    const formData = new FormData();
    formData.append('image', fs.createReadStream(req.file.path));

    const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
      headers: {
        ...formData.getHeaders(),
      },
    });

    fs.unlinkSync(req.file.path);
    res.json(response.data);
  } catch (error) {
    fs.unlinkSync(req.file.path);
    res.status(500).json({ error: 'Server error' });
  }
});

app.listen(3500, () => console.log('Server running on port 3500'));
