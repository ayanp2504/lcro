import os
import PyPDF2
from PyPDF2 import PdfReader

class PDFExtractor:
    def __init__(self, pdf_file_path, output_dir=None):
        """
        Initializes the PDFExtractor with the path to the PDF file and the output directory.

        Args:
            pdf_file_path (str): Path to the PDF file.
            output_dir (str): Directory to save extracted images (if any).
        """
        self.pdf_file_path = pdf_file_path
        self.output_dir = output_dir
        self.extracted_text = ""

    def extract(self, extract_images=True, extract_text=True):
        """
        Extracts text and/or images from the PDF file based on input parameters.

        Args:
            extract_images (bool): Flag to indicate whether to extract images (default: True).
            extract_text (bool): Flag to indicate whether to extract text (default: True).

        Returns:
            str: Extracted text if extract_text is True, else an empty string.
        """
        try:
            with open(self.pdf_file_path, 'rb') as file:
                pdf_reader = PdfReader(file)

                for page in pdf_reader.pages:
                    if extract_text:
                        text = page.extract_text()
                        if text:
                            self.extracted_text += text
                
                    if extract_images and self.output_dir:
                        self._extract_images_from_page(page)

                print(f"Extraction completed for: {self.pdf_file_path}")

        except Exception as e:
            print(f"An error occurred during extraction: {e}")

        return self.extracted_text

    def _extract_images_from_page(self, page):
        """
        Extracts images from a given PDF page and saves them to the output directory.

        Args:
            page: The page object from which to extract images.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        image_count = 0
        
        for image_file_object in page.images:
            image_path = os.path.join(self.output_dir, f"image_{image_count}_{image_file_object.name}")
            with open(image_path, "wb") as fp:
                fp.write(image_file_object.data)
                image_count += 1

        print(f"Extracted {image_count} images from a page.")

if __name__ == "__main__":
    # Example usage
    pdf_path = r"D:\path\to\your\pdf_file.pdf"  # Path to your PDF file
    output_directory = r"D:\path\to\output\directory"  # Desired output directory

    extractor = PDFExtractor(pdf_path, output_directory)
    extracted_text = extractor.extract()
    print("Extracted Text:\n", extracted_text)
