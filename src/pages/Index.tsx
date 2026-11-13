import { useEffect, useState } from "react";
import { FileUpload } from "@/components/FileUpload";
import { ImagePreview } from "@/components/ImagePreview";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

const Index = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [musicXml, setMusicXml] = useState<string | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Clean up object URLs
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (downloadUrl) URL.revokeObjectURL(downloadUrl);
    };
  }, [previewUrl, downloadUrl]);

  const handleFileSelect = (file: File) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);

    const url = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(url);
    setMusicXml(null);
    if (downloadUrl) URL.revokeObjectURL(downloadUrl);
    setDownloadUrl(null);
  };

  const handleClear = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (downloadUrl) URL.revokeObjectURL(downloadUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setMusicXml(null);
    setDownloadUrl(null);
  };

  const handleConvert = async () => {
    if (!selectedFile) {
      toast.error("Please upload a PNG sheet music file first.");
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const res = await fetch(`${import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000"}/api/omr`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const xmlText = await res.text();
      setMusicXml(xmlText);

      const blob = new Blob([xmlText], {
        type: "application/vnd.recordare.musicxml+xml",
      });
      const url = URL.createObjectURL(blob);
      if (downloadUrl) URL.revokeObjectURL(downloadUrl);
      setDownloadUrl(url);

      toast.success("MusicXML generated successfully!");
    } catch (err) {
      console.error(err);
      toast.error("Failed to convert image. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = () => {
    if (!downloadUrl || !selectedFile) return;
    const a = document.createElement("a");
    a.href = downloadUrl;
    a.download = `${
      selectedFile.name.replace(/\.[^.]+$/, "") || "measure"
    }.musicxml`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#fff6e5] to-[#ffe7c7] text-slate-900">
      {/* Top bar */}
      <header className="w-full border-b border-orange-100 bg-[#fff7e7]">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-[#ff9966] to-[#ff5e62] text-xl font-bold text-white">
              ♪
            </div>
            <div>
              <h1 className="text-lg font-semibold">Sheet Music Converter</h1>
              <p className="text-xs text-slate-500">PNG to MusicXML</p>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-10 space-y-10">
        {/* Hero text */}
        <section className="space-y-4 text-center">
          <h2 className="text-3xl font-bold md:text-4xl">
            Transform Sheet Music to{" "}
            <span className="text-[#ff7a5c]">Digital Format</span>
          </h2>
          <p className="mx-auto max-w-2xl text-slate-600">
            Upload your sheet music images and convert them to MusicXML format
            for easy editing and sharing with notation software.
          </p>
        </section>

        {/* Main content */}
        <section className="grid items-start gap-8 md:grid-cols-[minmax(0,2fr)_minmax(0,1.5fr)]">
          {/* LEFT COLUMN */}
          <div className="space-y-6">
            {/* Upload card */}
            <div className="rounded-3xl border-2 border-dashed border-[#f7c29b] bg-[#fffaf3] shadow-sm">
              <div className="p-8">
                <FileUpload
                  selectedFile={selectedFile}
                  onFileSelect={handleFileSelect}
                  onClear={handleClear}
                />
              </div>
            </div>

            {/* Image preview sits directly UNDER the file box */}
            {selectedFile && (
              <div className="rounded-3xl border border-orange-100 bg-white p-6 shadow-sm">
                <h3 className="mb-3 font-semibold">Preview</h3>
                <ImagePreview file={selectedFile} />
              </div>
            )}

            {/* Feature cards if NO file yet */}
            {!selectedFile && (
              <div className="grid gap-4 md:grid-cols-1">
                <div className="rounded-3xl bg-white p-5 shadow-sm">
                  <h3 className="mb-1 font-semibold">Easy Upload</h3>
                  <p className="text-sm text-slate-600">
                    Simply drag and drop your PNG sheet music files.
                  </p>
                </div>
                <div className="rounded-3xl bg-white p-5 shadow-sm">
                  <h3 className="mb-1 font-semibold">Smart Conversion</h3>
                  <p className="text-sm text-slate-600">
                    Our model recognizes a single whole note in a one-measure
                    PNG.
                  </p>
                </div>
                <div className="rounded-3xl bg-white p-5 shadow-sm">
                  <h3 className="mb-1 font-semibold">MusicXML Output</h3>
                  <p className="text-sm text-slate-600">
                    Download ready-to-use MusicXML files for your notation
                    software.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* RIGHT COLUMN: MusicXML preview + buttons */}
          <div className="space-y-6">
            <div className="space-y-4 rounded-3xl bg-gradient-to-b from-[#ff8f6b] to-[#ff6b4a] p-6 text-white shadow-lg">
              <h3 className="text-lg font-semibold">MusicXML Preview</h3>
              <p className="text-sm text-orange-100">
                Click convert to run the OMR model on your uploaded measure.
                The generated MusicXML will appear below and can be downloaded.
              </p>

              {/* MusicXML text preview */}
              <div className="mt-2 rounded-2xl bg-white/95 p-3 text-left text-xs text-slate-800 shadow-inner max-h-72 overflow-auto whitespace-pre-wrap font-mono">
                {musicXml
                  ? musicXml
                  : " "}
              </div>

              {/* Buttons */}
              <div className="mt-4 flex gap-3">
                <Button
                  className="flex-1 bg-white text-[#ff6b4a] hover:bg-orange-50 disabled:opacity-60"
                  onClick={handleConvert}
                  disabled={isLoading || !selectedFile}
                >
                  {isLoading ? "Converting..." : "Convert"}
                </Button>
                <Button
                  variant="outline"
                  className="flex-1 border-white/70 bg-transparent text-white hover:bg-white/10 disabled:opacity-50"
                  onClick={handleDownload}
                  disabled={!downloadUrl}
                >
                  Download
                </Button>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
};

export default Index;