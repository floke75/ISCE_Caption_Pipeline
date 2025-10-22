export async function copyToClipboard(text: string): Promise<boolean> {
  if (!text) {
    return false;
  }

  if (navigator.clipboard && navigator.clipboard.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (error) {
      console.warn('Clipboard write failed, falling back to selection', error);
    }
  }

  const textArea = document.createElement('textarea');
  textArea.value = text;
  textArea.style.position = 'fixed';
  textArea.style.opacity = '0';
  document.body.appendChild(textArea);
  textArea.focus({ preventScroll: true });
  textArea.select();

  try {
    const succeeded = document.execCommand('copy');
    return succeeded;
  } catch (error) {
    console.warn('document.execCommand copy failed', error);
    return false;
  } finally {
    document.body.removeChild(textArea);
  }
}
