import { FileUploader } from './components/FileUploader'
import './App.css'

function App() {
  return (
    <div className="App">
      <header className="app-header">
        <h1>File Uploader</h1>
        <p>Mobile-first drag and drop file upload</p>
      </header>
      <main>
        <FileUploader />
      </main>
    </div>
  )
}

export default App