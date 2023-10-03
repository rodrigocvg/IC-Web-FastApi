const fileInput = document.querySelector('#videoFileInput'); // Seletor do input de arquivo
const submitButton = document.querySelector('#submitButton'); // Seletor do botão de envio

submitButton.addEventListener('click', async () => {
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('video_file', file);

  try {
    const response = await axios.post('http://localhost:8000/process_video/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    console.log(response.data);
  } catch (error) {
    console.error(error);
  }
});